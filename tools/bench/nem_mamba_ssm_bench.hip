// Nemotron Bolt Mamba2 SSM scan microbench (gfx1151).
// Production MambaSsmKernel<32>: 128 blocks x 256 threads (one block per
// head, 16 heads/group), every thread runs the `tokens`-long serial
// recurrence and reloads bx/cc as 64 scalar broadcasts per token. Probe
// float4 state staging and a prefetch double buffer.
//
// tools/bench/build.sh nem_mamba_ssm_bench && /tmp/nem_mamba_ssm_bench [T...]

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr int kInner = 8192;
constexpr int kHeads = 128;
constexpr int kHd = 64;
constexpr int kState = 128;
constexpr int kGroups = 8;
constexpr int kXbc = kInner + 2 * kGroups * kState;  // 10240
constexpr int kProj = 2 * kInner + 2 * kGroups * kState + kHeads;  // 18560
constexpr int kBlock = 256;
constexpr int kQuads = kBlock / 4;
constexpr int kPer = 32;

inline __device__ float SiluD(float v) {
  return v / (1.0F + expf(-v));
}
inline __device__ float SoftplusD(float v) {
  return v > 20.0F ? v : log1pf(expf(v));
}

// A: production baseline (scalar broadcast loads of bx/cc).
__global__ void SsmBase(const float* __restrict__ zxbc,
                        const float* __restrict__ cx,
                        const float* __restrict__ a, const float* __restrict__ d,
                        const float* __restrict__ dtb, float* __restrict__ ssm_state,
                        float* __restrict__ y, int tokens) {
  const int tid = static_cast<int>(threadIdx.x);
  const int quad = blockIdx.x * kQuads + tid / 4;
  const int s = tid % 4;
  const int h = quad / kHd;
  const int g = h / (kHeads / kGroups);
  const float af = __ldg(a + h);
  const float df = __ldg(d + h);
  const float dtb_f = __ldg(dtb + h);
  const int st0 = s * kPer;

  float sp[kPer];
  float* spg = ssm_state + static_cast<int64_t>(quad) * kState + st0;
  for (int j = 0; j < kPer; ++j) sp[j] = spg[j];
  for (int t = 0; t < tokens; ++t) {
    const float* zt = zxbc + static_cast<int64_t>(t) * kProj;
    const float* cx_t = cx + static_cast<int64_t>(t) * kXbc;
    const float dt_sp = SoftplusD(zt[kProj - kInner / kHd + h] + dtb_f);
    const float xv = cx_t[quad] * dt_sp;
    const float dA = expf(dt_sp * af);
    const float* bx = cx_t + kInner + g * kState + st0;
    const float* cc = cx_t + kInner + (kGroups + g) * kState + st0;
    float sum = 0.0F;
    for (int j = 0; j < kPer; ++j) {
      const float ns = sp[j] * dA + bx[j] * xv;
      sp[j] = ns;
      sum += ns * cc[j];
    }
    sum += __shfl_xor_sync(~0ULL, sum, 1);
    sum += __shfl_xor_sync(~0ULL, sum, 2);
    const float zz = zt[quad];
    y[static_cast<int64_t>(t) * kInner + quad] = (sum + cx_t[quad] * df) * SiluD(zz);
  }
  for (int j = 0; j < kPer; ++j) spg[j] = sp[j];
}

// V: float4 state staging (same arithmetic order, j ascending).
__global__ void SsmVec4(const float* __restrict__ zxbc,
                        const float* __restrict__ cx,
                        const float* __restrict__ a, const float* __restrict__ d,
                        const float* __restrict__ dtb, float* __restrict__ ssm_state,
                        float* __restrict__ y, int tokens) {
  const int tid = static_cast<int>(threadIdx.x);
  const int quad = blockIdx.x * kQuads + tid / 4;
  const int s = tid % 4;
  const int h = quad / kHd;
  const int g = h / (kHeads / kGroups);
  const float af = __ldg(a + h);
  const float df = __ldg(d + h);
  const float dtb_f = __ldg(dtb + h);
  const int st0 = s * kPer;

  float sp[kPer];
  float* spg = ssm_state + static_cast<int64_t>(quad) * kState + st0;
  for (int j = 0; j < kPer; ++j) sp[j] = spg[j];
  for (int t = 0; t < tokens; ++t) {
    const float* zt = zxbc + static_cast<int64_t>(t) * kProj;
    const float* cx_t = cx + static_cast<int64_t>(t) * kXbc;
    const float dt_sp = SoftplusD(zt[kProj - kInner / kHd + h] + dtb_f);
    const float xv = cx_t[quad] * dt_sp;
    const float dA = expf(dt_sp * af);
    const float4* bx =
        reinterpret_cast<const float4*>(cx_t + kInner + g * kState + st0);
    const float4* cc =
        reinterpret_cast<const float4*>(cx_t + kInner + (kGroups + g) * kState +
                                        st0);
    float sum = 0.0F;
#pragma unroll
    for (int v = 0; v < kPer / 4; ++v) {
      const float4 b = bx[v];
      const float4 c = cc[v];
      float* s4 = sp + v * 4;
      const float n0 = s4[0] * dA + b.x * xv;
      const float n1 = s4[1] * dA + b.y * xv;
      const float n2 = s4[2] * dA + b.z * xv;
      const float n3 = s4[3] * dA + b.w * xv;
      s4[0] = n0; s4[1] = n1; s4[2] = n2; s4[3] = n3;
      sum += n0 * c.x;
      sum += n1 * c.y;
      sum += n2 * c.z;
      sum += n3 * c.w;
    }
    sum += __shfl_xor_sync(~0ULL, sum, 1);
    sum += __shfl_xor_sync(~0ULL, sum, 2);
    const float zz = zt[quad];
    y[static_cast<int64_t>(t) * kInner + quad] = (sum + cx_t[quad] * df) * SiluD(zz);
  }
  for (int j = 0; j < kPer; ++j) spg[j] = sp[j];
}

// P: float4 staging + manual prefetch of t+1 (unroll-2, order preserved).
__global__ void SsmPrefetch(const float* __restrict__ zxbc,
                            const float* __restrict__ cx,
                            const float* __restrict__ a,
                            const float* __restrict__ d,
                            const float* __restrict__ dtb,
                            float* __restrict__ ssm_state, float* __restrict__ y,
                            int tokens) {
  const int tid = static_cast<int>(threadIdx.x);
  const int quad = blockIdx.x * kQuads + tid / 4;
  const int s = tid % 4;
  const int h = quad / kHd;
  const int g = h / (kHeads / kGroups);
  const float af = __ldg(a + h);
  const float df = __ldg(d + h);
  const float dtb_f = __ldg(dtb + h);
  const int st0 = s * kPer;
  const int bc0 = kInner + g * kState + st0;
  const int cc0 = kInner + (kGroups + g) * kState + st0;

  float sp[kPer];
  float* spg = ssm_state + static_cast<int64_t>(quad) * kState + st0;
  for (int j = 0; j < kPer; ++j) sp[j] = spg[j];

  float4 b[8], c[8];
  const float4* bx0 = reinterpret_cast<const float4*>(cx + bc0);
  const float4* cc0p = reinterpret_cast<const float4*>(cx + cc0);
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    b[v] = bx0[v];
    c[v] = cc0p[v];
  }
  for (int t = 0; t < tokens; ++t) {
    const float* zt = zxbc + static_cast<int64_t>(t) * kProj;
    const float* cx_t = cx + static_cast<int64_t>(t) * kXbc;
    const float dt_sp = SoftplusD(zt[kProj - kInner / kHd + h] + dtb_f);
    const float xv = cx_t[quad] * dt_sp;
    const float dA = expf(dt_sp * af);
    float4 nb[8], nc[8];
    const int nxt = t + 1 < tokens ? kXbc : 0;
    const float4* nbx = reinterpret_cast<const float4*>(cx_t + nxt + bc0);
    const float4* ncc = reinterpret_cast<const float4*>(cx_t + nxt + cc0);
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      nb[v] = nbx[v];
      nc[v] = ncc[v];
    }
    float sum = 0.0F;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      float* s4 = sp + v * 4;
      const float n0 = s4[0] * dA + b[v].x * xv;
      const float n1 = s4[1] * dA + b[v].y * xv;
      const float n2 = s4[2] * dA + b[v].z * xv;
      const float n3 = s4[3] * dA + b[v].w * xv;
      s4[0] = n0; s4[1] = n1; s4[2] = n2; s4[3] = n3;
      sum += n0 * c[v].x;
      sum += n1 * c[v].y;
      sum += n2 * c[v].z;
      sum += n3 * c[v].w;
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      b[v] = nb[v];
      c[v] = nc[v];
    }
    sum += __shfl_xor_sync(~0ULL, sum, 1);
    sum += __shfl_xor_sync(~0ULL, sum, 2);
    const float zz = zt[quad];
    y[static_cast<int64_t>(t) * kInner + quad] = (sum + cx_t[quad] * df) * SiluD(zz);
  }
  for (int j = 0; j < kPer; ++j) spg[j] = sp[j];
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> toks;
  if (argc > 1) {
    for (int i = 1; i < argc; ++i) toks.push_back(atoi(argv[i]));
  } else {
    toks = {8192, 2048};
  }
  const int maxT = *std::max_element(toks.begin(), toks.end());
  float *zxbc, *cx, *y, *st0, *a, *d, *dtb;
  CHECK(hipMalloc(&zxbc, (size_t)maxT * kProj * 4));
  CHECK(hipMalloc(&cx, (size_t)maxT * kXbc * 4));
  CHECK(hipMalloc(&y, (size_t)maxT * kInner * 4));
  CHECK(hipMalloc(&st0, (size_t)kInner * kState * 4));
  CHECK(hipMalloc(&a, kHeads * 4));
  CHECK(hipMalloc(&d, kHeads * 4));
  CHECK(hipMalloc(&dtb, kHeads * 4));
  float* dummy = nullptr;
  CHECK(hipMalloc(&dummy, 4));
  {
    std::vector<float> h((size_t)maxT * kProj);
    for (size_t i = 0; i < h.size(); ++i)
      h[i] = std::sin(0.001F * (float)(i % 977)) * 0.5F;
    CHECK(hipMemcpy(zxbc, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(cx, h.data(), (size_t)maxT * kXbc * 4, hipMemcpyHostToDevice));
    std::vector<float> hd(kHeads);
    for (int i = 0; i < kHeads; ++i) hd[i] = -0.5F - 0.01F * i;
    CHECK(hipMemcpy(a, hd.data(), hd.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d, hd.data(), hd.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dtb, hd.data(), hd.size() * 4, hipMemcpyHostToDevice));
  }
  float* ya = nullptr;
  float* yv = nullptr;
  float* yp = nullptr;
  CHECK(hipMalloc(&ya, (size_t)maxT * kInner * 4));
  CHECK(hipMalloc(&yv, (size_t)maxT * kInner * 4));
  CHECK(hipMalloc(&yp, (size_t)maxT * kInner * 4));

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  const int grid = kInner * 4 / kBlock;

  auto launch = [&](int v, float* yout, int tokens) {
    if (v == 0)
      SsmBase<<<grid, kBlock, 0, stream>>>(zxbc, cx, a, d, dtb, st0, yout, tokens);
    else if (v == 1)
      SsmVec4<<<grid, kBlock, 0, stream>>>(zxbc, cx, a, d, dtb, st0, yout, tokens);
    else
      SsmPrefetch<<<grid, kBlock, 0, stream>>>(zxbc, cx, a, d, dtb, st0, yout,
                                               tokens);
    CHECK(hipGetLastError());
  };

  for (int t : toks) {
    const char* names[3] = {"base", "vec4", "prefetch"};
    float* ysel[3] = {ya, yv, yp};
    // Bit-equality check: same fresh zero state, y and final state equal.
    const size_t ybytes = (size_t)t * kInner * 4;
    std::vector<float> refY;
    std::vector<float> refState;
    for (int v = 0; v < 3; ++v) {
      CHECK(hipMemset(st0, 0, (size_t)kInner * kState * 4));
      CHECK(hipMemset(ysel[v], 0, ybytes));
      launch(v, ysel[v], t);
      CHECK(hipStreamSynchronize(stream));
      std::vector<float> hv(ybytes / 4);
      CHECK(hipMemcpy(hv.data(), ysel[v], ybytes, hipMemcpyDeviceToHost));
      std::vector<float> hs((size_t)kInner * kState);
      CHECK(hipMemcpy(hs.data(), st0, hs.size() * 4, hipMemcpyDeviceToHost));
      if (v == 0) {
        refY = std::move(hv);
        refState = std::move(hs);
      } else {
        printf("  tokens=%d %s == base: y %s, state %s\n", t, names[v],
               hv == refY ? "BITWISE" : "DIFF",
               hs == refState ? "BITWISE" : "DIFF");
      }
    }
    float best[3] = {1e30F, 1e30F, 1e30F};
    for (int it = 0; it < 7; ++it) {
      for (int v = 0; v < 3; ++v) {
        CHECK(hipEventRecord(e0, stream));
        launch(v, ysel[v], t);
        CHECK(hipEventRecord(e1, stream));
        CHECK(hipEventSynchronize(e1));
        float ms = 0.0F;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        if (ms < best[v]) best[v] = ms;
      }
    }
    printf("tokens=%d:", t);
    for (int v = 0; v < 3; ++v) printf("  %s=%.3f ms", names[v], best[v]);
    printf("\n");
  }
  return 0;
}
