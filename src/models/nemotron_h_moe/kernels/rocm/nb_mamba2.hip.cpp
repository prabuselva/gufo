#include "src/models/nemotron_h_moe/kernels/rocm/nb_mamba2.h"

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kBlock = 256;
constexpr int kQuads = kBlock / 4;  // (head, i) slots handled per block.

__device__ inline float SiluD(float x) {
  return x / (1.0F + expf(-x));
}

__device__ inline float SoftplusD(float x) {
  return x > 20.0F ? x : log1pf(expf(x));
}

// One thread per xBC channel; runs the causal convolutions of the whole
// token chunk sequentially, mirroring the oracle window shift exactly.
__global__ void MambaConvKernel(const float* __restrict__ zxbc,
                                const float* __restrict__ cw,
                                const float* __restrict__ cb,
                                float* __restrict__ conv_state,
                                float* __restrict__ cx, int inner, int taps,
                                int xbc, int proj, int tokens) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= xbc) {
    return;
  }
  float hist[3];
  for (int k = 0; k < taps - 1; ++k) {
    hist[k] = conv_state[static_cast<int64_t>(k) * xbc + ch];
  }
  const float w_new = cw[ch * taps + taps - 1];
  for (int t = 0; t < tokens; ++t) {
    const float nv = zxbc[static_cast<int64_t>(t) * proj + inner + ch];
    float acc = cb != nullptr ? cb[ch] : 0.0F;
    for (int k = 0; k < taps - 1; ++k) {
      acc += cw[ch * taps + k] * hist[k];
    }
    acc += w_new * nv;
    cx[static_cast<int64_t>(t) * xbc + ch] = SiluD(acc);
    for (int k = 0; k + 2 < taps; ++k) {
      hist[k] = hist[k + 1];
    }
    if (taps > 1) {
      hist[taps - 2] = nv;
    }
  }
  for (int k = 0; k < taps - 1; ++k) {
    conv_state[static_cast<int64_t>(k) * xbc + ch] = hist[k];
  }
}

// Thread group of 4 owns one (head, i) recurrent row; member s owns the
// state slice [s * PER, (s + 1) * PER]. The state lives in registers for the
// whole chunk and is written back once at the end.
template<int PER>
__global__ void MambaSsmKernel(
    const float* __restrict__ zxbc, const float* __restrict__ cx,
    const float* __restrict__ a, const float* __restrict__ d,
    const float* __restrict__ dtb, float* __restrict__ ssm_state,
    float* __restrict__ y, int inner, int hd, int state, int groups,
    int heads_per_group, int xbc, int proj, int tokens) {
  const int tid = static_cast<int>(threadIdx.x);
  const int quad = blockIdx.x * kQuads + tid / 4;
  const int s = tid % 4;
  const int h = quad / hd;
  const int g = h / heads_per_group;
  const float af = __ldg(a + h);
  const float df = __ldg(d + h);
  const float dtb_f = __ldg(dtb + h);
  const int st0 = s * PER;

  float sp[PER];
  float* spg = ssm_state + static_cast<int64_t>(quad) * state + st0;
  for (int j = 0; j < PER; ++j) {
    sp[j] = spg[j];
  }
  for (int t = 0; t < tokens; ++t) {
    const float* zt = zxbc + static_cast<int64_t>(t) * proj;
    const float* cx_t = cx + static_cast<int64_t>(t) * xbc;
    const float dt_sp = SoftplusD(zt[proj - inner / hd + h] + dtb_f);
    const float xv = cx_t[quad] * dt_sp;
    const float dA = expf(dt_sp * af);
    const float* bx = cx_t + inner + g * state + st0;
    const float* cc = cx_t + inner + (groups + g) * state + st0;
    float sum = 0.0F;
    for (int j = 0; j < PER; ++j) {
      const float ns = sp[j] * dA + bx[j] * xv;
      sp[j] = ns;
      sum += ns * cc[j];
    }
    // Butterfly over the four state slices keeps the total in all members
    // (summation order differs from the oracle; parity keeps a tolerance).
    sum += __shfl_xor_sync(~0ULL, sum, 1);
    sum += __shfl_xor_sync(~0ULL, sum, 2);
    const float zz = zt[quad];
    y[static_cast<int64_t>(t) * inner + quad] =
        (sum + cx_t[quad] * df) * SiluD(zz);
  }
  for (int j = 0; j < PER; ++j) {
    spg[j] = sp[j];
  }
}

}  // namespace

extern "C" int nbl_mamba_scan(const float* zxbc, const float* cw,
                              const float* cb, const float* a, const float* d,
                              const float* dtb, float* conv_state,
                              float* ssm_state, float* y, int inner, int heads,
                              int hd, int state, int groups, int taps, int xbc,
                              int proj, int tokens, hipStream_t stream) {
  if (!zxbc || !cw || !conv_state || !ssm_state || !y || inner <= 0 ||
      heads <= 0 || hd <= 0 || groups <= 0 || heads % groups != 0 ||
      state <= 0 || state % 4 != 0 || state / 4 > 64 || inner % 64 != 0 ||
      taps < 2 || taps > 4 || xbc < inner + 2 * groups * state ||
      proj < 2 * inner + 2 * groups * state + heads || tokens <= 0) {
    return -1;
  }
  float* cx = nullptr;
  const std::size_t cx_bytes =
      static_cast<std::size_t>(tokens) * xbc * sizeof(float);
  if (hipMallocAsync(&cx, cx_bytes, stream) != hipSuccess) {
    return -2;
  }
  MambaConvKernel<<<(xbc + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      zxbc, cw, cb, conv_state, cx, inner, taps, xbc, proj, tokens);
  if (hipGetLastError() != hipSuccess) {
    hipFreeAsync(cx, stream);
    return -3;
  }
  const int grid = inner * 4 / kBlock;
  const int heads_per_group = heads / groups;
  switch (state / 4) {
    case 1:
      MambaSsmKernel<1><<<grid, kBlock, 0, stream>>>(
          zxbc, cx, a, d, dtb, ssm_state, y, inner, hd, state, groups,
          heads_per_group, xbc, proj, tokens);
      break;
    case 2:
      MambaSsmKernel<2><<<grid, kBlock, 0, stream>>>(
          zxbc, cx, a, d, dtb, ssm_state, y, inner, hd, state, groups,
          heads_per_group, xbc, proj, tokens);
      break;
    case 4:
      MambaSsmKernel<4><<<grid, kBlock, 0, stream>>>(
          zxbc, cx, a, d, dtb, ssm_state, y, inner, hd, state, groups,
          heads_per_group, xbc, proj, tokens);
      break;
    case 8:
      MambaSsmKernel<8><<<grid, kBlock, 0, stream>>>(
          zxbc, cx, a, d, dtb, ssm_state, y, inner, hd, state, groups,
          heads_per_group, xbc, proj, tokens);
      break;
    case 16:
      MambaSsmKernel<16><<<grid, kBlock, 0, stream>>>(
          zxbc, cx, a, d, dtb, ssm_state, y, inner, hd, state, groups,
          heads_per_group, xbc, proj, tokens);
      break;
    case 32:
      MambaSsmKernel<32><<<grid, kBlock, 0, stream>>>(
          zxbc, cx, a, d, dtb, ssm_state, y, inner, hd, state, groups,
          heads_per_group, xbc, proj, tokens);
      break;
    default:
      MambaSsmKernel<64><<<grid, kBlock, 0, stream>>>(
          zxbc, cx, a, d, dtb, ssm_state, y, inner, hd, state, groups,
          heads_per_group, xbc, proj, tokens);
      break;
  }
  hipFreeAsync(cx, stream);
  return hipGetLastError() == hipSuccess ? 0 : -4;
}
