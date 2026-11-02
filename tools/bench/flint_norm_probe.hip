// opt-c7v-norm-quant probe: fuse Flint's AddResidual + RmsNorm + the tiled
// Q8_1 activation quantize into one wavefront-per-row-block kernel.
//
// Flint RmsNormKernel strides 256 threads over the row, so with cols % 256 == 0
// every output-loop iteration leaves warp w holding exactly the 32 elements of
// quantization block (iteration * 8 + w): the shipped per-wave quantizer's
// shape already matches the norm loop, and the quantization tree runs on the
// same FP32 value the norm stored. Every q byte, scale and sidecar value must
// be bit-identical to [AddResidual][RmsNorm][QuantizeActivation4ToQ8_1], and
// the normed row and updated residual stream bit-identical to the chain.
#if defined(ENGINE_ENABLE_HIP)

#include "src/models/flint/kernels/rocm/kernels.hip.cpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

// kernels.hip.cpp dispatches Attention to attention_wmma.hip.cpp; the probe
// never launches attention, so provide a link stub in its place.
namespace gufo::models::flint::rocm {
bool LaunchWmmaAttention(const float* q, const __half* k_cache,
                         const __half* v_cache, const float* sinks, float scale,
                         std::uint32_t pos0, std::uint32_t batch,
                         std::uint32_t window, std::uint32_t n_head,
                         std::uint32_t n_kv, std::uint32_t kv_row_elements,
                         float* out, hipStream_t stream) {
  std::fprintf(stderr, "Attention not supported in this probe\n");
  std::abort();
}
}  // namespace gufo::models::flint::rocm

namespace probe {

using gufo::hip::kQ8ActScaleOffset;
using gufo::hip::kQ8ActTileBytes;
using gufo::hip::kQ8ActTileTokens;
using gufo::hip::Q8ActSumSidecar;
using gufo::hip::Q8ActTiledBytes;
using gufo::hip::StoreQ8ActLane;

constexpr int kThreads = 256;

// Verbatim copy of the shipped C7u chain quantizer.
template<bool StoreActivationSum>
__global__ void ChainQuantKernel(const float* __restrict__ x,
                                 void* __restrict__ y, std::size_t batch,
                                 std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t num_chunks = num_blocks >> 2u;
  const std::size_t wave =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane = threadIdx.x & 31u;
  if (wave >= batch * num_chunks) {
    return;
  }

  const std::size_t tok = wave / num_chunks;
  const std::size_t chunk = wave % num_chunks;
  const std::size_t blk = chunk * 4u + (lane >> 3u);
  const std::size_t sub = lane & 7u;
  const std::size_t tl = tok % kQ8ActTileTokens;
  const std::size_t tt = tok / kQ8ActTileTokens;

  const float4 v4 =
      reinterpret_cast<const float4*>(x)[(tok * k) / 4u + chunk * 32u + lane];
  const float elems[4] = {v4.x, v4.y, v4.z, v4.w};
  float max_abs = fmaxf(fmaxf(fabsf(elems[0]), fabsf(elems[1])),
                        fmaxf(fabsf(elems[2]), fabsf(elems[3])));
  for (int off = 4; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }

  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;

  std::int8_t* tile = gufo::hip::Q8ActTile(y, num_blocks, tt, blk);
  int qsum = 0;
#pragma unroll
  for (std::size_t j = 0; j < 4u; ++j) {
    const std::size_t pos = sub * 4u + j;
    const std::int8_t q = static_cast<std::int8_t>(roundf(elems[j] * id));
    tile[((pos >> 4u) * 256u) + (tl * 16u) + (pos & 15u)] = q;
    qsum = qsum + static_cast<int>(q);
    if (sub == 0U && j == 0U) {
      *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                (tl * sizeof(float))) = d;
    }
  }
  if constexpr (StoreActivationSum) {
    for (int off = 4; off > 0; off >>= 1) {
      qsum += __shfl_xor(qsum, off);
    }
    if (sub == 0U) {
      float* sums = Q8ActSumSidecar(y, batch, num_blocks * 32);
      sums[(((tt * num_blocks) + blk) * kQ8ActTileTokens) + tl] =
          d * static_cast<float>(qsum);
    }
  }
}

// Verbatim copy of the shipped tail zero.
template<bool StoreActivationSum>
__global__ void TailZeroKernel(void* __restrict__ y, std::size_t batch,
                               std::size_t num_blocks) {
  const std::size_t tail = batch % kQ8ActTileTokens;
  if (tail == 0) {
    return;
  }
  const std::size_t tt = batch / kQ8ActTileTokens;
  const std::size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t lanes = kQ8ActTileTokens - tail;
  if (idx >= num_blocks * lanes) {
    return;
  }
  const std::size_t kb = idx / lanes;
  const std::size_t tl = tail + (idx % lanes);
  std::int8_t* tile = gufo::hip::Q8ActTile(y, num_blocks, tt, kb);
  reinterpret_cast<gufo::hip::int32x4_t*>(tile + (tl * 16))[0] = {};
  reinterpret_cast<gufo::hip::int32x4_t*>(tile + 256 + (tl * 16))[0] = {};
  *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
      0.0F;
  if constexpr (StoreActivationSum) {
    float* sums = Q8ActSumSidecar(y, batch, num_blocks * 32);
    sums[(((tt * num_blocks) + kb) * kQ8ActTileTokens) + tl] = 0.0F;
  }
}

template<bool StoreActivationSum>
void LaunchTailZero(void* q8, std::size_t batch, std::size_t k,
                    hipStream_t stream) {
  const std::size_t tail = batch % kQ8ActTileTokens;
  if (tail == 0) {
    return;
  }
  const std::size_t num_blocks = k / 32;
  const std::size_t work = num_blocks * (kQ8ActTileTokens - tail);
  TailZeroKernel<StoreActivationSum>
      <<<static_cast<unsigned int>((work + 255) / 256), 256, 0, stream>>>(
          q8, batch, num_blocks);
}

__device__ __forceinline__ float ProbeBlockReduceSum(float v, float* sm) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(__activemask(), v, off);
  }
  if (lane == 0) {
    sm[warp] = v;
  }
  __syncthreads();
  const int warps = blockDim.x / 32;
  float total = 0.0F;
  if (threadIdx.x == 0) {
    for (int i = 0; i < warps; ++i) {
      total += sm[i];
    }
    sm[0] = total;
  }
  __syncthreads();
  return sm[0];
}

// Fused candidate: in-place residual add, RMSNorm (identical tree and
// expressions to the shipped RmsNormKernel), and the per-wave Q8_1 quantize of
// the normed value. Requires cols % 256 == 0 for the quantize path.
template<bool AddResidual, bool Quantize>
__global__ void FusedNormQuantKernel(float* __restrict__ x,
                                     const float* __restrict__ y,
                                     const float* __restrict__ gamma,
                                     float eps, float* __restrict__ out,
                                     void* __restrict__ q8, std::size_t batch,
                                     std::uint32_t cols) {
  __shared__ float sm[kThreads / 32];
  float* __restrict__ xr =
      x + static_cast<std::size_t>(blockIdx.x) * cols;
  const float* __restrict__ yr =
      AddResidual ? y + static_cast<std::size_t>(blockIdx.x) * cols : nullptr;
  float* __restrict__ orow = out + static_cast<std::size_t>(blockIdx.x) * cols;
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < cols; i += kThreads) {
    float v = xr[i];
    if constexpr (AddResidual) {
      v += yr[i];
      xr[i] = v;
    }
    ss += v * v;
  }
  ss = ProbeBlockReduceSum(ss, sm);
  const float scale = 1.0F / sqrtf(ss / static_cast<float>(cols) + eps);
  if constexpr (!Quantize) {
    for (std::uint32_t i = threadIdx.x; i < cols; i += kThreads) {
      orow[i] = xr[i] * scale * gamma[i];
    }
    return;
  }
  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t warp = threadIdx.x >> 5u;
  const std::size_t num_blocks = cols / 32u;
  const std::size_t tok = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < cols; i += kThreads) {
    const float val = xr[i] * scale * gamma[i];
    orow[i] = val;
    const std::size_t blk =
        (static_cast<std::size_t>(i) / kThreads) * (kThreads / 32u) + warp;
    if (blk >= num_blocks) {
      continue;
    }
    float max_abs = fabsf(val);
    for (int off = 16; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    }
    const float d = max_abs / 127.0F;
    const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
    const std::int8_t q = static_cast<std::int8_t>(roundf(val * id));
    StoreQ8ActLane<true>(q8, batch, num_blocks, tok, blk, lane, d, q);
  }
}

void LaunchChainQuant(const float* normed, void* q8, std::size_t batch,
                      std::size_t k, hipStream_t stream) {
  const std::size_t waves = batch * (k / 128u);
  const std::size_t grid = (waves * 32u + kThreads - 1) / kThreads;
  ChainQuantKernel<true><<<static_cast<unsigned int>(grid), kThreads, 0,
                           stream>>>(normed, q8, batch, k);
  LaunchTailZero<true>(q8, batch, k, stream);
}

template<bool AddResidual, bool Quantize>
void LaunchFused(float* x, const float* y, const float* gamma, float eps,
                 float* out, void* q8, std::size_t batch, std::size_t cols,
                 hipStream_t stream) {
  FusedNormQuantKernel<AddResidual, Quantize>
      <<<static_cast<unsigned int>(batch), kThreads, 0, stream>>>(
          x, y, gamma, eps, out, q8, batch,
          static_cast<std::uint32_t>(cols));
  if constexpr (Quantize) {
    LaunchTailZero<true>(q8, batch, cols, stream);
  }
}

float TimeIt(std::function<void()> fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  hipEvent_t a, b;
  hipEventCreate(&a);
  hipEventCreate(&b);
  hipEventRecord(a);
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  hipEventRecord(b);
  hipEventSynchronize(b);
  float ms = 0.0F;
  hipEventElapsedTime(&ms, a, b);
  hipEventDestroy(a);
  hipEventDestroy(b);
  return ms / iters;
}

std::size_t Q8Bytes(std::size_t batch, std::size_t k) {
  const std::size_t tiles = (batch + kQ8ActTileTokens - 1) / kQ8ActTileTokens;
  return Q8ActTiledBytes(batch, k) +
         tiles * (k / 32u) * kQ8ActTileTokens * sizeof(float);
}

int RunCase(std::size_t batch, std::size_t k, float eps) {
  const std::size_t n = batch * k;
  std::vector<float> hx(n), hy(n), hgam(k), hxn(n);
  std::mt19937 rng(12345 + static_cast<unsigned>(batch));
  std::uniform_real_distribution<float> uni(-1.0F, 1.0F);
  for (std::size_t i = 0; i < n; ++i) {
    hx[i] = uni(rng);
    hy[i] = uni(rng);
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (i < 512) {  // exercise zero and extreme blocks
      hx[i] = 0.0F;
    } else if (i < 640) {
      hx[i] = (i & 1) ? 1e30F : -1e30F;
    }
  }
  std::uniform_real_distribution<float> gam(0.5F, 1.5F);
  for (std::size_t i = 0; i < k; ++i) {
    hgam[i] = gam(rng);
  }

  float *dx, *dy, *dg, *dn_a, *dn_f, *dx_f;
  void* dq_a;
  void* dq_f;
  const std::size_t qb = Q8Bytes(batch, k);
  const std::size_t guard = 256;
  hipMalloc(&dx, n * sizeof(float));
  hipMalloc(&dy, n * sizeof(float));
  hipMalloc(&dg, k * sizeof(float));
  hipMalloc(&dn_a, n * sizeof(float));
  hipMalloc(&dn_f, n * sizeof(float));
  hipMalloc(&dx_f, n * sizeof(float));
  hipMalloc(&dq_a, qb + guard);
  hipMalloc(&dq_f, qb + guard);
  hipMemcpy(dx, hx.data(), n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dx_f, hx.data(), n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dy, hy.data(), n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dg, hgam.data(), k * sizeof(float), hipMemcpyHostToDevice);
  hipMemset(dn_a, 0x5A, n * sizeof(float));
  hipMemset(dn_f, 0x5A, n * sizeof(float));
  hipMemset(dq_a, 0xA5, qb + guard);
  hipMemset(dq_f, 0xA5, qb + guard);

  // Chain: in-place AddResidual on the norm input, RmsNorm, C7u quantize.
  auto chain = [&]() {
    gufo::models::flint::rocm::AddResidual(dx, dy,
                                           static_cast<std::uint32_t>(batch),
                                           static_cast<std::uint32_t>(k), 0);
    gufo::models::flint::rocm::RmsNorm(dx, dg, eps, dn_a,
                                       static_cast<std::uint32_t>(batch),
                                       static_cast<std::uint32_t>(k), 0);
    LaunchChainQuant(dn_a, dq_a, batch, k, 0);
  };
  // Fused variant used in production site B: residual + norm + quant.
  auto fused = [&]() {
    LaunchFused<true, true>(dx_f, dy, dg, eps, dn_f, dq_f, batch, k, 0);
  };

  chain();
  fused();
  hipDeviceSynchronize();

  std::vector<float> hxa(n), hxb(n), hna(n), hnb(n);
  std::vector<unsigned char> hqa(qb), hqf(qb), guard_a(guard), guard_f(guard);
  hipMemcpy(hxa.data(), dx, n * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(hxb.data(), dx_f, n * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(hna.data(), dn_a, n * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(hnb.data(), dn_f, n * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(hqa.data(), dq_a, qb, hipMemcpyDeviceToHost);
  hipMemcpy(hqf.data(), dq_f, qb, hipMemcpyDeviceToHost);
  hipMemcpy(guard_a.data(),
            static_cast<unsigned char*>(dq_a) + qb, guard,
            hipMemcpyDeviceToHost);
  hipMemcpy(guard_f.data(),
            static_cast<unsigned char*>(dq_f) + qb, guard,
            hipMemcpyDeviceToHost);

  int bad = 0;
  auto report = [&](const char* what, long idx, double a, double b) {
    if (bad < 8) {
      printf("  MISMATCH %s[%ld]: %g vs %g\n", what, idx, a, b);
    }
    ++bad;
  };
  for (std::size_t i = 0; i < n; ++i) {
    if (std::memcmp(&hxa[i], &hxb[i], sizeof(float)) != 0) {
      report("x_res", static_cast<long>(i), hxa[i], hxb[i]);
    }
    if (std::memcmp(&hna[i], &hnb[i], sizeof(float)) != 0) {
      report("normed", static_cast<long>(i), hna[i], hnb[i]);
    }
  }
  for (std::size_t i = 0; i < qb; ++i) {
    if (hqa[i] != hqf[i]) {
      report("q8", static_cast<long>(i), hqa[i], hqf[i]);
    }
  }
  for (std::size_t i = 0; i < guard; ++i) {
    bad += (guard_a[i] != 0xA5 ? 1 : 0) + (guard_f[i] != 0xA5 ? 1 : 0);
  }

  const float ms_chain = TimeIt(chain, 5, 60);
  const float ms_fused = TimeIt(fused, 5, 60);
  const double traffic_mb = 3.0 * n * sizeof(float) / 1e6;
  printf("batch=%zu k=%zu mismatches=%d chain=%.4f ms fused=%.4f ms "
         "gain=%.1f%% (~%.0f MB/pass)\n",
         batch, k, bad, ms_chain, ms_fused,
         100.0 * (ms_chain - ms_fused) / ms_chain, traffic_mb);

  // Site A variant: norm + quantize with no residual (fresh inputs).
  hipMemcpy(dx, hx.data(), n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dx_f, hx.data(), n * sizeof(float), hipMemcpyHostToDevice);
  auto chain_nr = [&]() {
    gufo::models::flint::rocm::RmsNorm(dx, dg, eps, dn_a,
                                       static_cast<std::uint32_t>(batch),
                                       static_cast<std::uint32_t>(k), 0);
    LaunchChainQuant(dn_a, dq_a, batch, k, 0);
  };
  auto fused_nr = [&]() {
    LaunchFused<false, true>(dx_f, dy, dg, eps, dn_f, dq_f, batch, k, 0);
  };
  chain_nr();
  fused_nr();
  hipDeviceSynchronize();
  hipMemcpy(hna.data(), dn_a, n * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(hnb.data(), dn_f, n * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(hqa.data(), dq_a, qb, hipMemcpyDeviceToHost);
  hipMemcpy(hqf.data(), dq_f, qb, hipMemcpyDeviceToHost);
  int bad_nr = 0;
  for (std::size_t i = 0; i < n; ++i) {
    bad_nr += std::memcmp(&hna[i], &hnb[i], sizeof(float)) != 0 ? 1 : 0;
  }
  for (std::size_t i = 0; i < qb; ++i) {
    bad_nr += hqa[i] != hqf[i] ? 1 : 0;
  }
  const float ms_chain_nr = TimeIt(chain_nr, 5, 60);
  const float ms_fused_nr = TimeIt(fused_nr, 5, 60);
  printf("        no-res: mismatches=%d chain=%.4f ms fused=%.4f ms "
         "gain=%.1f%%\n",
         bad_nr, ms_chain_nr, ms_fused_nr,
         100.0 * (ms_chain_nr - ms_fused_nr) / ms_chain_nr);
  bad += bad_nr;

  hipFree(dx);
  hipFree(dy);
  hipFree(dg);
  hipFree(dn_a);
  hipFree(dn_f);
  hipFree(dx_f);
  hipFree(dq_a);
  hipFree(dq_f);
  return bad != 0 ? 1 : 0;
}

}  // namespace probe

int main() {
  int rc = 0;
  for (const std::size_t batch : {512u, 1000u, 1024u, 2048u, 2560u, 3072u,
                                  4096u, 6144u, 8192u}) {
    rc |= probe::RunCase(batch, 2048, 1e-6F);
  }
  printf("%s\n", rc == 0 ? "ALL BIT-IDENTICAL" : "FAILURES PRESENT");
  return rc;
}

#else
int main() {
  printf("ENGINE_ENABLE_HIP not defined\n");
  return 0;
}
#endif
