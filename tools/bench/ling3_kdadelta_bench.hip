// Feasibility probe: KdaDelta decode-kernel micro variants (ling3 lever
// 5.9). Production KdaDeltaKernel launches heads(=32)x128 threads and has
// each thread own one d=128 state row, streaming it twice through global
// memory with 32-b-thread-strided scalar accesses (avg 43 us vs a ~16.7 us
// DRAM roofline for the 2 MB r+w). This probe reuses the REAL production
// kernel as V0 and measures bit-safe variants at production shapes
// (heads=32, d=128): V1 vectorizes both row loops with float4 elementwise
// ops kept in ascending index order; V2 additionally keeps the whole row
// in registers, storing the final row once (production stores damped and
// final and reloads between the two dependent loops). D3 is an FP32
// accumulator attribution run: NOT bit-safe, reported only to price the
// double accumulate. Every variant is bitwise-compared against V0's
// state and out on identical inputs.
#include "src/models/ling3_0/kernels/rocm/kernels.hip.cpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define CU(x)                                                        \
  do {                                                               \
    hipError_t e_ = (x);                                             \
    if (e_ != hipSuccess) {                                          \
      std::fprintf(stderr, "HIP %s @%d: %s\n", #x, __LINE__,         \
                   hipGetErrorString(e_));                           \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

namespace gufo::models::ling3_0::rocm {
namespace {

constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kD = 128;
constexpr std::uint32_t kInner = kHeads * kD;
constexpr std::uint32_t kRing = 64;  // 64 x 2 MB state ring > L2
constexpr int kReps = 2;

// Bit-safe row-owned variants. Element order within each thread matches
// the production loop exactly; only load/store width and register
// residency change.
template <int kRegRow>
__global__ void KdaDeltaVariant(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ f,
    const float* __restrict__ beta, const float* __restrict__ a,
    const float* __restrict__ norm_w, const float* __restrict__ g,
    float* __restrict__ state, float* __restrict__ out, float eps,
    float gate_lb) {
  __shared__ double smem[8];
  __shared__ float s_k[kD];
  __shared__ float s_q[kD];
  __shared__ float s_u[kD];
  __shared__ float s_o[kD];
  __shared__ float s_decay[kD];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t j = threadIdx.x;
  const std::size_t hd = static_cast<std::size_t>(h) * kD;
  s_k[j] = k[hd + j];
  s_q[j] = q[hd + j];
  s_decay[j] = ::expf(gate_lb * SigmoidD(a[h] * f[hd + j]));
  __syncthreads();

  float* S = state + static_cast<std::size_t>(h) * kD * kD;
  float* row = S + static_cast<std::size_t>(j) * kD;
  const float4* r4 = reinterpret_cast<const float4*>(row);
  float4* w4 = reinterpret_cast<float4*>(row);
  double acc = 0.0;
  float4 reg[kRegRow ? kD / 4U : 1];
  if constexpr (kRegRow != 0) {
    // Register-resident row: 32 float4 = 128 floats, loaded once.
#pragma unroll
    for (std::uint32_t t = 0; t < kD / 4U; ++t) {
      reg[t] = r4[t];
    }
    float* re = reinterpret_cast<float*>(reg);
    for (std::uint32_t i = 0; i < kD; ++i) {
      re[i] *= s_decay[i];
      acc += static_cast<double>(re[i]) * static_cast<double>(s_k[i]);
    }
  } else {
#pragma unroll
    for (std::uint32_t t = 0; t < kD / 4U; ++t) {
      float4 r = r4[t];
      r.x *= s_decay[4U * t];
      r.y *= s_decay[4U * t + 1U];
      r.z *= s_decay[4U * t + 2U];
      r.w *= s_decay[4U * t + 3U];
      w4[t] = r;
      acc += static_cast<double>(r.x) * static_cast<double>(s_k[4U * t]);
      acc += static_cast<double>(r.y) * static_cast<double>(s_k[4U * t + 1U]);
      acc += static_cast<double>(r.z) * static_cast<double>(s_k[4U * t + 2U]);
      acc += static_cast<double>(r.w) * static_cast<double>(s_k[4U * t + 3U]);
    }
  }
  s_u[j] = static_cast<float>(acc);
  __syncthreads();

  const float b = SigmoidD(beta[h]);
  const float delta = (v[hd + j] - s_u[j]) * b;
  acc = 0.0;
  if constexpr (kRegRow != 0) {
    float* re = reinterpret_cast<float*>(reg);
#pragma unroll
    for (std::uint32_t i = 0; i < kD; ++i) {
      re[i] += delta * s_k[i];
      acc += static_cast<double>(re[i]) * static_cast<double>(s_q[i]);
    }
    float4* wr = reinterpret_cast<float4*>(row);
#pragma unroll
    for (std::uint32_t t = 0; t < kD / 4U; ++t) {
      wr[t] = reg[t];
    }
  } else {
#pragma unroll
    for (std::uint32_t t = 0; t < kD / 4U; ++t) {
      float4 r = r4[t];
      r.x += delta * s_k[4U * t];
      r.y += delta * s_k[4U * t + 1U];
      r.z += delta * s_k[4U * t + 2U];
      r.w += delta * s_k[4U * t + 3U];
      w4[t] = r;
      acc += static_cast<double>(r.x) * static_cast<double>(s_q[4U * t]);
      acc += static_cast<double>(r.y) * static_cast<double>(s_q[4U * t + 1U]);
      acc += static_cast<double>(r.z) * static_cast<double>(s_q[4U * t + 2U]);
      acc += static_cast<double>(r.w) * static_cast<double>(s_q[4U * t + 3U]);
    }
  }
  s_o[j] = static_cast<float>(acc);
  __syncthreads();

  const double o = static_cast<double>(s_o[j]);
  const double ss = BlockSumDouble(o * o, smem);
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(ss / static_cast<double>(kD)) + eps);
  out[hd + j] = s_o[j] * scale * norm_w[j] * SigmoidD(g[hd + j]);
}

/// D3 attribution only: production loop shape with an FP32 accumulator.
/// Not bit-safe; priced solely to show what the double accumulate costs.
__global__ void KdaDeltaF32Acc(const float* __restrict__ q,
                               const float* __restrict__ k,
                               const float* __restrict__ v,
                               const float* __restrict__ f,
                               const float* __restrict__ beta,
                               const float* __restrict__ a,
                               const float* __restrict__ norm_w,
                               const float* __restrict__ g,
                               float* __restrict__ state,
                               float* __restrict__ out, float eps,
                               float gate_lb) {
  __shared__ double smem[8];
  __shared__ float s_k[kD];
  __shared__ float s_q[kD];
  __shared__ float s_u[kD];
  __shared__ float s_o[kD];
  __shared__ float s_decay[kD];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t j = threadIdx.x;
  const std::size_t hd = static_cast<std::size_t>(h) * kD;
  s_k[j] = k[hd + j];
  s_q[j] = q[hd + j];
  s_decay[j] = ::expf(gate_lb * SigmoidD(a[h] * f[hd + j]));
  __syncthreads();
  float* S = state + static_cast<std::size_t>(h) * kD * kD;
  float* row = S + static_cast<std::size_t>(j) * kD;
  float acc = 0.0F;
  for (std::uint32_t i = 0; i < kD; ++i) {
    row[i] *= s_decay[i];
    acc += row[i] * s_k[i];
  }
  s_u[j] = acc;
  __syncthreads();
  const float b = SigmoidD(beta[h]);
  const float delta = (v[hd + j] - s_u[j]) * b;
  acc = 0.0F;
  for (std::uint32_t i = 0; i < kD; ++i) {
    row[i] += delta * s_k[i];
    acc += row[i] * s_q[i];
  }
  s_o[j] = acc;
  __syncthreads();
  const double o = static_cast<double>(s_o[j]);
  const double ss = BlockSumDouble(o * o, smem);
  const float scale =
      1.0F / ::sqrtf(static_cast<float>(ss / static_cast<double>(kD)) + eps);
  out[hd + j] = s_o[j] * scale * norm_w[j] * SigmoidD(g[hd + j]);
}

/// Bitwise diff counter against the golden buffers.
__global__ void DiffKernel(const uint4* golden_s, const uint4* got_s,
                           std::size_t n_s, const int* golden_o,
                           const int* got_o, std::size_t n_o,
                           std::uint32_t* mismatches) {
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n_s + n_o; i += stride) {
    bool bad;
    if (i < n_s) {
      bad = golden_s[i].x != got_s[i].x || golden_s[i].y != got_s[i].y ||
            golden_s[i].z != got_s[i].z || golden_s[i].w != got_s[i].w;
    } else {
      bad = golden_o[i - n_s] != got_o[i - n_s];
    }
    if (bad) {
      atomicAdd(mismatches, 1U);
    }
  }
}

// Deterministic in-range inputs so expf/sigmoid paths stay realistic.
__global__ void FillKernel(float* p, std::size_t n, unsigned seed) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned x = static_cast<unsigned>(i) * 2654435761U + seed;
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  const float f = static_cast<float>(x & 0xFFFFU) / 32768.0F;  // [0,2)
  p[i] = f - 1.0F;
}

float* Alloc(float val_seed = 0U, std::size_t n = 0, unsigned seed = 1U) {
  float* p = nullptr;
  CU(hipMalloc(&p, n * sizeof(float)));
  if (val_seed != 0U) {
    FillKernel<<<(n + 255U) / 256U, 256>>>(p, n, seed);
  }
  return p;
}

void Launch(int variant, const float* q, const float* k, const float* v,
            const float* f, const float* beta, const float* a,
            const float* norm_w, const float* g, float* state, float* out,
            float eps, float gate_lb) {
  switch (variant) {
    case 0:
      KdaDeltaKernel<<<kHeads, kD, 0, nullptr>>>(q, k, v, f, beta, a, norm_w,
                                                  g, state, out, kD, eps,
                                                  gate_lb);
      break;
    case 1:
      KdaDeltaVariant<0><<<kHeads, kD, 0, nullptr>>>(q, k, v, f, beta, a,
                                                     norm_w, g, state, out,
                                                     eps, gate_lb);
      break;
    case 2:
      KdaDeltaVariant<1><<<kHeads, kD, 0, nullptr>>>(q, k, v, f, beta, a,
                                                     norm_w, g, state, out,
                                                     eps, gate_lb);
      break;
    case 3:
      KdaDeltaF32Acc<<<kHeads, kD, 0, nullptr>>>(q, k, v, f, beta, a, norm_w,
                                                 g, state, out, eps, gate_lb);
      break;
  }
}

}  // namespace
}  // namespace gufo::models::ling3_0::rocm

using namespace gufo::models::ling3_0::rocm;

int main() {
  const float* q = Alloc(1, kInner, 11);
  const float* k = Alloc(1, kInner, 22);
  const float* v = Alloc(1, kInner, 33);
  const float* f = Alloc(1, kInner, 44);
  const float* g = Alloc(1, kInner, 55);
  const float* norm_w = Alloc(1, kD, 66);
  const float* beta = Alloc(1, kHeads, 77);
  const float* a = Alloc(1, kHeads, 88);
  float* state[kRing];
  for (std::uint32_t l = 0; l < kRing; ++l) {
    state[l] = Alloc(1, static_cast<std::size_t>(kHeads) * kD * kD, 100 + l);
  }
  float* out = Alloc(1, kInner);
  float* golden_state = Alloc(1, static_cast<std::size_t>(kHeads) * kD * kD);
  float* golden_out = Alloc(1, kInner);
  std::uint32_t* mismatches = nullptr;
  CU(hipMalloc(&mismatches, sizeof(std::uint32_t)));

  const float eps = 1e-6F;
  const float gate_lb = 1.0F;
  constexpr std::size_t kStateN = static_cast<std::size_t>(kHeads) * kD * kD;

  // Golden pass: production kernel on one fresh fill; keep the same input
  // state aside so every variant starts from identical bytes.
  float* base_state = Alloc(1, kStateN);
  FillKernel<<<(kStateN + 255U) / 256U, 256>>>(base_state, kStateN, 500);
  FillKernel<<<(kStateN + 255U) / 256U, 256>>>(state[0], kStateN, 500);
  Launch(0, q, k, v, f, beta, a, norm_w, g, state[0], out, eps, gate_lb);
  CU(hipMemcpy(golden_state, state[0], kStateN * sizeof(float),
               hipMemcpyDeviceToDevice));
  CU(hipMemcpy(golden_out, out, kInner * sizeof(float), hipMemcpyDeviceToDevice));

  const char* names[4] = {"V0 production ", "V1 float4     ",
                          "V2 reg-row    ", "D3 f32-acc(diag)"};
  for (int variant = 1; variant < 4; ++variant) {
    CU(hipMemset(mismatches, 0, sizeof(std::uint32_t)));
    CU(hipMemcpy(state[1], base_state, kStateN * sizeof(float),
                 hipMemcpyDeviceToDevice));
    Launch(variant, q, k, v, f, beta, a, norm_w, g, state[1], out, eps,
           gate_lb);
    DiffKernel<<<1024, 256>>>(reinterpret_cast<const uint4*>(golden_state),
                              reinterpret_cast<const uint4*>(state[1]),
                              kStateN / 4U,
                              reinterpret_cast<const int*>(golden_out),
                              reinterpret_cast<const int*>(out), kInner / 4U,
                              mismatches);
    std::uint32_t m = 0;
    CU(hipMemcpy(&m, mismatches, sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    std::printf("%s : %s\n  bitwise mismatch words: %u\n", names[variant],
                m == 0U ? "BIT-EXACT vs V0" : "DIFFERS", m);
    std::fflush(stdout);
  }

  // Timing: best of rounds, one full 64-layer ring pass per rep.
  hipEvent_t ev0, ev1;
  CU(hipEventCreate(&ev0));
  CU(hipEventCreate(&ev1));
  double med[4] = {1e30, 1e30, 1e30, 1e30};
  for (int variant = 0; variant < 4; ++variant) {
    for (int rep = 0; rep < kReps; ++rep) {
      Launch(variant, q, k, v, f, beta, a, norm_w, g, state[0], out, eps,
             gate_lb);
    }
  }
  CU(hipDeviceSynchronize());
  for (int round = 0; round < 6; ++round) {
    for (int variant = 0; variant < 4; ++variant) {
      CU(hipEventRecord(ev0, nullptr));
      for (int rep = 0; rep < kReps; ++rep) {
        for (std::uint32_t l = 0; l < kRing; ++l) {
          Launch(variant, q, k, v, f, beta, a, norm_w, g, state[l], out, eps,
                 gate_lb);
        }
      }
      CU(hipEventRecord(ev1, nullptr));
      CU(hipEventSynchronize(ev1));
      float ms = 0.0f;
      CU(hipEventElapsedTime(&ms, ev0, ev1));
      med[variant] =
          std::min(med[variant],
                   static_cast<double>(ms) * 1000.0 / (kReps * kRing));
    }
  }
  for (int variant = 0; variant < 4; ++variant) {
    std::printf("%s : %7.2f us/launch", names[variant], med[variant]);
    if (variant > 0) {
      std::printf("  (%+.1f%% vs V0)", (med[variant] / med[0] - 1) * 100);
    }
    std::printf("\n");
  }
  return 0;
}
