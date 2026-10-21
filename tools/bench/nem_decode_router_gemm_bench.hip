// Nemotron Bolt decode router GEMM bench #2 (gfx1151): out[1,n_exp] = x w^T,
// F32, n_exp 240/368/496, k = 4096, one token. The adopted thread-per-output
// kernel (commit 4ae98bee) measured 44 us with one weight buffer (L2-hot);
// decode profiled at ~355 us per launch because the per-lane row chains are
// memory-latency bound (24 GB/s) and weights are L2-cold across 40 layers.
// Probe bit-identical variants: float4 direct streaming, and an LDS-staged
// design where all 256 block threads cooperatively float4-load 8-row x 128-k
// tiles while the 8 row adders accumulate ascending-k serially, reproducing
// the production fold order exactly. Weight sets rotate over 48 buffers
// (>> 32 MiB L2) to reproduce decode reality; an L2-hot sweep inflated the
// n_exp=240 case by 3x. Sweep over prefetch depth: d2/d3 win (d4+ spills
// cost more than the extra latency hidden); d2 adopted in production:
// L2-cold 3.9/6.0/7.9 MB weight matrices go 358/357/377 us -> 58/73/92 us
// (4.1-6.2x), all bitwise identical.
//
//   tools/bench/build.sh nem_decode_router_gemm_bench && \
//   /tmp/nem_decode_router_gemm_bench

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#define CHECK(cmd)                                                    \
  do {                                                                \
    hipError_t err__ = (cmd);                                         \
    if (err__ != hipSuccess) {                                        \
      fprintf(stderr, "HIP error %s at %s:%d\n",                      \
              hipGetErrorString(err__), __FILE__, __LINE__);          \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

namespace {

constexpr int kK = 4096;
constexpr int kBlock = 256;
constexpr int kRows = 8;       // rows per block in the staged variant
constexpr int kStageK = 128;   // k-width of one cooperative LDS tile

// Production GemmF32Kernel (nb_ops.hip.cpp) verbatim: thread per output,
// scalar ascending-k chain.
__global__ void RowScalar(const float* __restrict__ x,
                          const float* __restrict__ w,
                          float* __restrict__ out, int n_out, int k) {
  const int t = blockIdx.y;
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_out) {
    return;
  }
  const float* xr = x + static_cast<int64_t>(t) * k;
  const float* wr = w + static_cast<int64_t>(r) * k;
  float acc = 0.0F;
  for (int i = 0; i < k; ++i) {
    acc += xr[i] * wr[i];
  }
  out[static_cast<int64_t>(t) * n_out + r] = acc;
}

// Same ownership, float4 loads; FMAs stay in ascending-k order.
__global__ void RowF4(const float* __restrict__ x, const float* __restrict__ w,
                      float* __restrict__ out, int n_out, int k) {
  const int t = blockIdx.y;
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_out) {
    return;
  }
  const float* xr = x + static_cast<int64_t>(t) * k;
  const float* wr = w + static_cast<int64_t>(r) * k;
  float acc = 0.0F;
  for (int i = 0; i < k; i += 4) {
    const float4 xv = *reinterpret_cast<const float4*>(xr + i);
    const float4 wv = *reinterpret_cast<const float4*>(wr + i);
    acc += xv.x * wv.x;
    acc += xv.y * wv.y;
    acc += xv.z * wv.z;
    acc += xv.w * wv.w;
  }
  out[static_cast<int64_t>(t) * n_out + r] = acc;
}

// Staged variant: one block owns kRows rows. All 256 threads cooperatively
// float4-load a kRows x kStageK tile (512 B contiguous per warp), then the
// first kRows threads fold their row from LDS in ascending k. Same per-row
// FMA order as the scalar kernel, so results are bitwise identical; the
// serial chain now runs at FMA + LDS latency instead of HBM latency.
__global__ void RowLds(const float* __restrict__ x, const float* __restrict__ w,
                       float* __restrict__ out, int n_out, int k) {
  const int t = blockIdx.y;
  const int r0 = blockIdx.x * kRows;
  const int tid = static_cast<int>(threadIdx.x);
  __shared__ float ws[kRows][kStageK + 1];
  const float* xr = x + static_cast<int64_t>(t) * k;

  const bool adder = (tid < kRows) && (r0 + tid < n_out);
  float acc = 0.0F;
  const int g = tid / 32;
  const int kl = (tid % 32) * 4;
  for (int kc = 0; kc < k; kc += kStageK) {
    const bool loader = (r0 + g) < n_out;
    float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (loader) {
      v = *reinterpret_cast<const float4*>(
          w + static_cast<int64_t>(r0 + g) * k + kc + kl);
    }
    ws[g][kl] = v.x;
    ws[g][kl + 1] = v.y;
    ws[g][kl + 2] = v.z;
    ws[g][kl + 3] = v.w;
    __syncthreads();
    if (adder) {
      const float* wr = ws[tid];
      const float* xrk = xr + kc;
#pragma unroll
      for (int p = 0; p < kStageK / 4; ++p) {
        const float4 xv = *reinterpret_cast<const float4*>(xrk + 4 * p);
        acc += xv.x * wr[4 * p];
        acc += xv.y * wr[4 * p + 1];
        acc += xv.z * wr[4 * p + 2];
        acc += xv.w * wr[4 * p + 3];
      }
    }
    __syncthreads();
  }
  if (adder) {
    out[static_cast<int64_t>(t) * n_out + r0 + tid] = acc;
  }
}

// Ring-buffer staged variant with parameterized prefetch depth: DEPTH tiles
// are in flight per pass so the row chain overlaps DEPTH HBM round trips.
template<int kStage, int DEPTH>
__global__ void RowLdsRing(const float* __restrict__ x,
                           const float* __restrict__ w,
                           float* __restrict__ out, int n_out, int k) {
  const int t = blockIdx.y;
  const int r0 = blockIdx.x * kRows;
  const int tid = static_cast<int>(threadIdx.x);
  __shared__ float ws[DEPTH][kRows][kStage + 1];
  const float* xr = x + static_cast<int64_t>(t) * k;

  const bool adder = (tid < kRows) && (r0 + tid < n_out);
  float acc = 0.0F;
  const int g = tid / 32;
  const int kl = (tid % 32) * 4;
  const int loader_ok = (r0 + g) < n_out ? 1 : 0;
  const float* wrow = w + static_cast<int64_t>(r0 + g) * k + kl;
  const int npass = k / kStage;
  float4 v[DEPTH];
  v[0] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
#pragma unroll
  for (int d = 0; d < DEPTH; ++d) {
    v[d] = (loader_ok && npass > d)
               ? *reinterpret_cast<const float4*>(wrow + d * kStage)
               : v[0];
  }
  for (int p = 0; p < npass; ++p) {
    const int cur = p % DEPTH;
    ws[cur][g][kl] = v[0].x;
    ws[cur][g][kl + 1] = v[0].y;
    ws[cur][g][kl + 2] = v[0].z;
    ws[cur][g][kl + 3] = v[0].w;
    __syncthreads();
#pragma unroll
    for (int d = 0; d + 1 < DEPTH; ++d) {
      v[d] = v[d + 1];
    }
    if (p + DEPTH < npass && loader_ok) {
      v[DEPTH - 1] = *reinterpret_cast<const float4*>(wrow + (p + DEPTH) * kStage);
    }
    if (adder) {
      const float* wr = ws[cur][tid];
      const float* xrk = xr + p * kStage;
#pragma unroll
      for (int q = 0; q < kStage / 4; ++q) {
        const float4 xv = *reinterpret_cast<const float4*>(xrk + 4 * q);
        acc += xv.x * wr[4 * q];
        acc += xv.y * wr[4 * q + 1];
        acc += xv.z * wr[4 * q + 2];
        acc += xv.w * wr[4 * q + 3];
      }
    }
    __syncthreads();
  }
  if (adder) {
    out[static_cast<int64_t>(t) * n_out + r0 + tid] = acc;
  }
}

struct Shape {
  const char* name;
  int n_exp;
};

uint32_t lcg(uint32_t& s) {
  s = s * 1664525u + 1013904223u;
  return s >> 8;
}

void run_case(const Shape& sh, int reps) {
  const int n_sets = 48;  // total weight footprint >> L2 for every shape
  const int64_t set_bytes =
      static_cast<int64_t>(sh.n_exp) * kK * static_cast<int64_t>(4);
  float* w = nullptr;
  float* x = nullptr;
  float* out_a = nullptr;
  float* out_b = nullptr;
  CHECK(hipMalloc(&w, static_cast<size_t>(set_bytes) * n_sets));
  CHECK(hipMalloc(&x, static_cast<size_t>(kK) * 4));
  CHECK(hipMalloc(&out_a, static_cast<size_t>(sh.n_exp) * 4));
  CHECK(hipMalloc(&out_b, static_cast<size_t>(sh.n_exp) * 4));
  std::vector<float> hw(static_cast<size_t>(set_bytes) * n_sets / 4);
  std::vector<float> hx(kK);
  uint32_t seed = 0x9abcu;
  for (auto& v : hw) {
    v = (static_cast<float>(lcg(seed)) / 8388608.0F) - 1.0F;
  }
  for (auto& v : hx) {
    v = (static_cast<float>(lcg(seed)) / 8388608.0F) - 1.0F;
  }
  CHECK(hipMemcpy(w, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));

  const dim3 block(kBlock, 1);
  auto launch_prod = [&](int set, float* out) {
    const dim3 grid(
        static_cast<unsigned>((sh.n_exp + kBlock - 1) / kBlock), 1);
    RowScalar<<<grid, block>>>(x, w + static_cast<int64_t>(set) * set_bytes / 4,
                               out, sh.n_exp, kK);
  };
  auto launch_f4 = [&](int set, float* out) {
    const dim3 grid(
        static_cast<unsigned>((sh.n_exp + kBlock - 1) / kBlock), 1);
    RowF4<<<grid, block>>>(x, w + static_cast<int64_t>(set) * set_bytes / 4,
                           out, sh.n_exp, kK);
  };
  auto launch_lds = [&](int set, float* out) {
    const dim3 grid(static_cast<unsigned>((sh.n_exp + kRows - 1) / kRows), 1);
    RowLds<<<grid, block>>>(x, w + static_cast<int64_t>(set) * set_bytes / 4,
                            out, sh.n_exp, kK);
  };
  const dim3 ring_grid(static_cast<unsigned>((sh.n_exp + kRows - 1) / kRows), 1);
  auto launch_ring = [&](auto depth_tag, int set, float* out) {
    constexpr int kDepth = decltype(depth_tag)::value;
    RowLdsRing<kStageK, kDepth><<<ring_grid, block>>>(
        x, w + static_cast<int64_t>(set) * set_bytes / 4, out, sh.n_exp, kK);
  };
  using D2 = std::integral_constant<int, 2>;
  using D3 = std::integral_constant<int, 3>;
  using D4 = std::integral_constant<int, 4>;
  using D5 = std::integral_constant<int, 5>;
  using D6 = std::integral_constant<int, 6>;
  auto l_d2 = [&](int s, float* o) { launch_ring(D2{}, s, o); };
  auto l_d3 = [&](int s, float* o) { launch_ring(D3{}, s, o); };
  auto l_d4 = [&](int s, float* o) { launch_ring(D4{}, s, o); };
  auto l_d5 = [&](int s, float* o) { launch_ring(D5{}, s, o); };
  auto l_d6 = [&](int s, float* o) { launch_ring(D6{}, s, o); };

  CHECK(hipMemset(out_a, 0xFF, static_cast<size_t>(sh.n_exp) * 4));
  launch_prod(0, out_a);
  CHECK(hipDeviceSynchronize());
  std::vector<float> href(sh.n_exp), hnew(sh.n_exp);
  CHECK(hipMemcpy(href.data(), out_a, href.size() * 4, hipMemcpyDeviceToHost));
  auto bitwise = [&](auto launch) {
    CHECK(hipMemset(out_b, 0xFF, static_cast<size_t>(sh.n_exp) * 4));
    launch(0, out_b);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(
        hipMemcpy(hnew.data(), out_b, hnew.size() * 4, hipMemcpyDeviceToHost));
    return std::memcmp(href.data(), hnew.data(), href.size() * 4) == 0 ? 1 : 0;
  };
  const int eq_f4 = bitwise(launch_f4);
  const int eq_lds = bitwise(launch_lds);
  const int eq_d2 = bitwise(l_d2);
  const int eq_d3 = bitwise(l_d3);
  const int eq_d4 = bitwise(l_d4);
  const int eq_d5 = bitwise(l_d5);
  const int eq_d6 = bitwise(l_d6);

  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  auto bench = [&](auto launch) {
    for (int i = 0; i < 3; ++i) launch(0, out_b);
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    for (int i = 0; i < reps; ++i) {
      launch(i % n_sets, (i % 2 == 0) ? out_b : out_a);
    }
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float ms = 0.0F;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    return static_cast<double>(ms) * 1000.0 / reps;
  };
  const double us_prod = bench(launch_prod);
  const double g = static_cast<double>(set_bytes) / 1e3;  // GB x us per launch
  const double us_d[] = {bench(l_d2), bench(l_d3), bench(l_d4), bench(l_d5),
                         bench(l_d6)};
  const int eq_d[] = {eq_d2, eq_d3, eq_d4, eq_d5, eq_d6};
  printf("%-8s n_exp=%-4d | prod %7.2f us (%5.1f GB/s) |", sh.name, sh.n_exp,
         us_prod, g / us_prod);
  for (int i = 0; i < 5; ++i) {
    printf(" d%d %7.2f (%5.1f GB/s) %5.2fx eq%d |", i + 2, us_d[i],
           g / us_d[i], us_prod / us_d[i], eq_d[i]);
  }
  printf(" base_eq %d %d\n", eq_f4, eq_lds);
  CHECK(hipFree(w));
  CHECK(hipFree(x));
  CHECK(hipFree(out_a));
  CHECK(hipFree(out_b));
}

}  // namespace

int main() {
  int warp = 0;
  CHECK(hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, 0));
  if (warp != 32) {
    fprintf(stderr, "expected warpSize 32, got %d\n", warp);
    return 1;
  }
  const Shape shapes[] = {{"bolt", 240}, {"bolt", 368}, {"max", 496}};
  for (const Shape& sh : shapes) {
    run_case(sh, 400);
  }
  return 0;
}
