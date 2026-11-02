// opt-c10 probe for the Flint QkEpilogue stage (kernels.hip.cpp, open lever
// "QkEpilogue ~111 ms per pp8192 profile: not yet analyzed"). The pp2048
// timed pass runs it at ~575 us/layer vs a ~170 us modelled DRAM floor
// (Q r/w fp32 + K r/o fp32/half + V r fp32 + half cache rows): one block per
// (token, head) with 128 threads makes it latency- and SFU-bound, not
// bandwidth-bound.
//
// Control: the shipped rocm::QkEpilogue from libgufo_flint_kernels.a.
// Variants are copies of the shipped kernel body that keep every per-element
// FP operation, reduction tree and rounding identical (verified bit for bit
// on random data, rope on and off) while:
//   R   - one block walks R tokens (row loop, identical reduce pattern per
//         row), cutting block/scheduler overhead 4-8x;
//   H   - hoists the per-thread powf(freq), gamma load and head_dim
//         constants out of the row loop (same powf call, same value);
//   SC  - replaces cosf/sinf with sincosf after a full-domain bitwise audit
//         over every (pos, d, theta) the model can hit and random angles.
//
// Build: tools/bench/build.sh flint_qk_bench
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                                     \
  do {                                                                  \
    const hipError_t err__ = (call);                                    \
    if (err__ != hipSuccess) {                                          \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                  hipGetErrorString(err__), __FILE__, __LINE__);        \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {

__device__ __forceinline__ float BlockReduceSum(float v, float* sm) {
  // Exact copy of the shipped helper (kernels.hip.cpp), 128-thread blocks.
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

// Verbatim copy of the shipped control (kernels.hip.cpp QkEpilogueKernel,
// HEAD at opt-c10) so the bench needs no archive link.
__global__ void QkEpilogueKernelCtl(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    float eps, float theta, int rope, std::uint32_t pos0, std::uint32_t n_head,
    std::uint32_t n_kv, std::uint32_t head_dim, __half* __restrict__ k_cache,
    __half* __restrict__ v_cache, std::uint32_t kv_row) {
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t h = blockIdx.x;
  const bool is_q = h < n_head;
  const std::uint32_t d = threadIdx.x;
  float* __restrict__ x =
      is_q ? q + (static_cast<std::size_t>(t) * n_head + h) * head_dim
           : k +
                 (static_cast<std::size_t>(t) * n_kv + (h - n_head)) * head_dim;
  __shared__ float sm_head[128];
  __shared__ float sm_part[8];
  const float xv = d < head_dim ? x[d] : 0.0F;
  float ss = xv * xv;
  ss = BlockReduceSum(ss, sm_part);
  const float scale = 1.0F / sqrtf(ss / static_cast<float>(head_dim) + eps);
  const float* gamma = is_q ? q_gamma : k_gamma;
  sm_head[d] = xv * scale * gamma[d];
  __syncthreads();
  const std::uint32_t pos = pos0 + t;
  if (rope != 0 && d < head_dim / 2) {
    const float freq = powf(
        theta, -2.0F * static_cast<float>(d) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(pos) * freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float a = sm_head[d];
    const float b = sm_head[d + head_dim / 2];
    sm_head[d] = a * c - b * s;
    sm_head[d + head_dim / 2] = a * s + b * c;
  }
  __syncthreads();
  if (is_q) {
    if (d < head_dim) {
      x[d] = sm_head[d];
    }
    return;
  }
  const std::uint32_t kvh = h - n_head;
  __half* __restrict__ krow =
      k_cache + static_cast<std::size_t>(pos) * kv_row + kvh * head_dim;
  const float* __restrict__ vrow =
      v + (static_cast<std::size_t>(t) * n_kv + kvh) * head_dim;
  __half* __restrict__ vwrite =
      v_cache + static_cast<std::size_t>(pos) * kv_row + kvh * head_dim;
  if (d < head_dim) {
    krow[d] = __float2half_rn(sm_head[d]);
    vwrite[d] = __float2half_rn(vrow[d]);
  }
}

// Shipped body, walked R tokens per block. `hoist` computes the per-thread
// freq/gamma once per block instead of once per row (identical value).
template <int R, bool hoist, bool sincos>
__global__ void QkEpilogueVar(
    float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ q_gamma, const float* __restrict__ k_gamma,
    float eps, float theta, int rope, std::uint32_t pos0, std::uint32_t n_head,
    std::uint32_t n_kv, std::uint32_t head_dim, __half* __restrict__ k_cache,
    __half* __restrict__ v_cache, std::uint32_t kv_row,
    std::uint32_t batch) {
  const std::uint32_t h = blockIdx.x;
  const bool is_q = h < n_head;
  const std::uint32_t d = threadIdx.x;
  __shared__ float sm_head[128];
  __shared__ float sm_part[8];
  const std::uint32_t t0 = blockIdx.y * R;

  float* __restrict__ xbase =
      is_q ? q + static_cast<std::size_t>(h) * head_dim
           : k + static_cast<std::size_t>(h - n_head) * head_dim;
  const std::size_t x_row =
      is_q ? static_cast<std::size_t>(n_head) * head_dim
           : static_cast<std::size_t>(n_kv) * head_dim;
  const float* __restrict__ gamma = is_q ? q_gamma : k_gamma;

  float freq = 0.0F;
  float gam = 0.0F;
  if (hoist) {
    gam = gamma[d];
    if (rope != 0 && d < head_dim / 2) {
      freq = powf(theta, -2.0F * static_cast<float>(d) /
                               static_cast<float>(head_dim));
    }
  }

  for (int i = 0; i < R; ++i) {
    const std::uint32_t t = t0 + i;
    if (t >= batch) {
      break;  // block-uniform: never skips a barrier
    }
    float* __restrict__ x = xbase + static_cast<std::size_t>(t) * x_row;
    const float xv = d < head_dim ? x[d] : 0.0F;
    float ss = xv * xv;
    ss = BlockReduceSum(ss, sm_part);
    const float scale = 1.0F / sqrtf(ss / static_cast<float>(head_dim) + eps);
    sm_head[d] = xv * scale * (hoist ? gam : gamma[d]);
    __syncthreads();
    const std::uint32_t pos = pos0 + t;
    if (rope != 0 && d < head_dim / 2) {
      const float f =
          hoist ? freq
                : powf(theta, -2.0F * static_cast<float>(d) /
                                  static_cast<float>(head_dim));
      const float angle = static_cast<float>(pos) * f;
      float c;
      float s;
      if constexpr (sincos) {
        sincosf(angle, &s, &c);
      } else {
        c = cosf(angle);
        s = sinf(angle);
      }
      const float a = sm_head[d];
      const float b = sm_head[d + head_dim / 2];
      sm_head[d] = a * c - b * s;
      sm_head[d + head_dim / 2] = a * s + b * c;
    }
    __syncthreads();
    if (is_q) {
      if (d < head_dim) {
        x[d] = sm_head[d];
      }
      continue;
    }
    const std::uint32_t kvh = h - n_head;
    __half* __restrict__ krow =
        k_cache + static_cast<std::size_t>(pos) * kv_row + kvh * head_dim;
    const float* __restrict__ vrow =
        v + (static_cast<std::size_t>(t) * n_kv + kvh) * head_dim;
    __half* __restrict__ vwrite =
        v_cache + static_cast<std::size_t>(pos) * kv_row + kvh * head_dim;
    if (d < head_dim) {
      krow[d] = __float2half_rn(sm_head[d]);
      vwrite[d] = __float2half_rn(vrow[d]);
    }
    // Shipped kernel leaves sm_head dirty between rows exactly like this
    // loop does: every lane rewrites its slot before use, so reuse is safe.
  }
}

struct Xorshift {
  std::uint64_t s = 88172645463325252ULL;
  std::uint32_t Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<std::uint32_t>(s >> 32);
  }
  float Uniform() { return static_cast<float>(Next()) / 4294967296.0F; }
};

std::vector<float> MakeRows(std::size_t n, Xorshift& rng) {
  std::vector<float> r(n);
  for (auto& f : r) {
    f = (rng.Uniform() - 0.5F) * 4.0F;
  }
  return r;
}

struct Case {
  std::uint32_t batch;
  std::uint32_t n_head;
  std::uint32_t n_kv;
  std::uint32_t head_dim;
  std::uint32_t pos0;
  float theta;
  bool rope;
};

// Runs one case: fills device state per call, launches control and variant
// onto separate buffer copies and compares bits, then times repeats.
template <int R, bool hoist, bool sincos>
void Bench(const Case& c, float eps, int reps) {
  const std::size_t hd = c.head_dim;
  const std::size_t qn = static_cast<std::size_t>(c.batch) * c.n_head * hd;
  const std::size_t kn = static_cast<std::size_t>(c.batch) * c.n_kv * hd;
  const std::size_t kvrow = static_cast<std::size_t>(c.n_kv) * hd;
  const std::size_t cache_n =
      (static_cast<std::size_t>(c.pos0) + c.batch + 8) * kvrow;
  Xorshift rng;
  std::vector<float> q0 = MakeRows(qn, rng), k0 = MakeRows(kn, rng),
                     v0 = MakeRows(kn, rng), qg = MakeRows(hd, rng),
                     kg = MakeRows(hd, rng);
  qg = MakeRows(hd, rng);  // positive gammas for stable norms
  for (auto& f : qg) f = rng.Uniform() + 0.5F;
  kg.assign(hd, 1.0F);
  for (auto& f : kg) f = rng.Uniform() + 0.5F;

  float *dq, *dk, *dv, *dqg, *dkg;
  __half *dkc, *dvc;
  CHECK(hipMalloc(&dq, qn * 4));
  CHECK(hipMalloc(&dk, kn * 4));
  CHECK(hipMalloc(&dv, kn * 4));
  CHECK(hipMalloc(&dqg, hd * 4));
  CHECK(hipMalloc(&dkg, hd * 4));
  CHECK(hipMalloc(&dkc, cache_n * 2));
  CHECK(hipMalloc(&dvc, cache_n * 2));

  auto Reset = [&]() {
    CHECK(hipMemcpy(dq, q0.data(), qn * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dk, k0.data(), kn * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dv, v0.data(), kn * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dqg, qg.data(), hd * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dkg, kg.data(), hd * 4, hipMemcpyHostToDevice));
    CHECK(hipMemset(dkc, 0, cache_n * 2));
    CHECK(hipMemset(dvc, 0, cache_n * 2));
  };

  const dim3 grid(c.n_head + c.n_kv, c.batch);
  const dim3 grid_r(c.n_head + c.n_kv, (c.batch + R - 1) / R);
  auto LaunchCtl = [&]() {
    QkEpilogueKernelCtl<<<grid, 128>>>(dq, dk, dv, dqg, dkg, eps, c.theta,
                                       c.rope ? 1 : 0, c.pos0, c.n_head,
                                       c.n_kv, c.head_dim, dkc, dvc,
                                       static_cast<std::uint32_t>(kvrow));
  };
  auto LaunchVar = [&]() {
    QkEpilogueVar<R, hoist, sincos><<<grid_r, 128>>>(
        dq, dk, dv, dqg, dkg, eps, c.theta, c.rope ? 1 : 0, c.pos0, c.n_head,
        c.n_kv, c.head_dim, dkc, dvc, static_cast<std::uint32_t>(kvrow),
        c.batch);
  };
  (void)grid;

  // Correctness: control vs variant over every buffer, bit for bit.
  std::vector<float> qc(qn), kc(kn), vc(kn);
  std::vector<__half> cc(cache_n), cv(cache_n);
  Reset();
  LaunchCtl();
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(qc.data(), dq, qn * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(kc.data(), dk, kn * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(vc.data(), dv, kn * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(cc.data(), dkc, cache_n * 2, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(cv.data(), dvc, cache_n * 2, hipMemcpyDeviceToHost));
  Reset();
  LaunchVar();
  CHECK(hipDeviceSynchronize());
  std::size_t mism = 0;
  {
    std::vector<float> a(qn);
    CHECK(hipMemcpy(a.data(), dq, qn * 4, hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < qn; ++i) {
      mism += std::memcmp(&a[i], &qc[i], 4) != 0;
    }
    CHECK(hipMemcpy(a.data(), dk, kn * 4, hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < kn; ++i) {
      mism += std::memcmp(&a[i], &kc[i], 4) != 0;
    }
    std::vector<__half> b(cache_n);
    CHECK(hipMemcpy(b.data(), dkc, cache_n * 2, hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < cache_n; ++i) {
      mism += std::memcmp(&b[i], &cc[i], 2) != 0;
    }
    CHECK(hipMemcpy(b.data(), dvc, cache_n * 2, hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < cache_n; ++i) {
      mism += std::memcmp(&b[i], &cv[i], 2) != 0;
    }
  }

  // Timing: back-to-back launches on the same buffers (cache rewrites are
  // idempotent for timing purposes). Model: q r+w, k r+w (fp32), v read,
  // half-precision k+v cache row writes per token.
  const double bytes =
      qn * 4.0 * 2.0 + kn * 4.0 * 2.0 + kn * 4.0 + kn * 2.0 * 2.0;
  {
    Reset();
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    for (int i = 0; i < 3; ++i) LaunchCtl();
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    for (int i = 0; i < reps; ++i) LaunchCtl();
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    ms /= reps;
    std::printf("  %-18s %9.1f us  %7.1f GB/s  mismatches %zu\n", "ctl shipped",
                ms * 1e3F, bytes / (ms * 1e-3) / 1e9, mism);
    CHECK(hipEventDestroy(e0));
    CHECK(hipEventDestroy(e1));
  }
  {
    Reset();
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    for (int i = 0; i < 3; ++i) LaunchVar();
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    for (int i = 0; i < reps; ++i) LaunchVar();
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    ms /= reps;
    char nm[64];
    std::snprintf(nm, sizeof nm, "R%d%s%s", R, hoist ? "+H" : "",
                  sincos ? "+SC" : "");
    std::printf("  %-18s %9.1f us  %7.1f GB/s  mismatches %zu\n", nm, ms * 1e3F,
                bytes / (ms * 1e-3) / 1e9, mism);
    CHECK(hipEventDestroy(e0));
    CHECK(hipEventDestroy(e1));
  }

  CHECK(hipFree(dq));
  CHECK(hipFree(dk));
  CHECK(hipFree(dv));
  CHECK(hipFree(dqg));
  CHECK(hipFree(dkg));
  CHECK(hipFree(dkc));
  CHECK(hipFree(dvc));
}

// Bitwise audit of sincosf(angle,&s,&c) vs sinf/cosf over every angle the
// rope can produce: d in [0,64), pos in [0, 131072), theta candidates, plus
// the exact float products pos*freq for a dense pos sweep.
__device__ __forceinline__ bool BitsEq(float a, float b) {
  return __builtin_bit_cast(std::uint32_t, a) ==
         __builtin_bit_cast(std::uint32_t, b);
}

__global__ void SincosAudit(int* bad, float theta, std::uint32_t pos_offset,
                            std::uint32_t max_pos) {
  const std::uint32_t d = blockIdx.x;
  const std::uint32_t pos =
      pos_offset + blockIdx.y * blockDim.x + threadIdx.x;
  if (d >= 64 || pos >= max_pos) return;
  const float f = powf(theta, -2.0F * static_cast<float>(d) / 128.0F);
  const float angle = static_cast<float>(pos) * f;
  const float s = sinf(angle);
  const float c = cosf(angle);
  float s2, c2;
  sincosf(angle, &s2, &c2);
  if (!BitsEq(s, s2) || !BitsEq(c, c2)) {
    atomicAdd(bad, 1);
  }
}

void AuditSincos() {
  int* dbad;
  CHECK(hipMalloc(&dbad, 4));
  const float thetas[] = {10000.0F, 500000.0F, 1000000.0F, 5000000.0F};
  for (float th : thetas) {
    CHECK(hipMemset(dbad, 0, 4));
    const dim3 grid(64, 1024);
    for (std::uint32_t base = 0; base < 131072; base += 32768) {
      SincosAudit<<<grid, 32>>>(dbad, th, base, base + 32768);
    }
    int host_bad = -1;
    CHECK(hipMemcpy(&host_bad, dbad, 4, hipMemcpyDeviceToHost));
    std::printf("  sincos audit theta=%.0f: %d mismatches\n", th, host_bad);
  }
  CHECK(hipFree(dbad));
}

}  // namespace

int main(int argc, char** argv) {
  int reps = 50;
  if (argc > 1) reps = std::atoi(argv[1]);
  CHECK(hipSetDevice(0));
  const float eps = 1e-6F;
  AuditSincos();
  const Case cases[] = {
      {64, 32, 4, 128, 0, 1000000.0F, true},
      {2048, 32, 4, 128, 0, 1000000.0F, true},
      {2048, 32, 4, 128, 0, 1000000.0F, false},
      {4096, 32, 4, 128, 2048, 1000000.0F, true},
  };
  for (const Case& c : cases) {
    std::printf("case B=%u pos0=%u rope=%d theta=%.0f\n", c.batch, c.pos0,
                c.rope ? 1 : 0, c.theta);
    Bench<2, false, false>(c, eps, reps);
    Bench<4, false, false>(c, eps, reps);
    Bench<4, true, false>(c, eps, reps);
    Bench<8, true, false>(c, eps, reps);
    Bench<4, true, true>(c, eps, reps);
    Bench<8, true, true>(c, eps, reps);
  }
  return 0;
}
