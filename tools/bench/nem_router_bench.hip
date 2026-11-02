// Nemotron Bolt MoE router GEMM microbench (gfx1151): out[M,512] = x[M,4096] w^T.
// Production GemmF32Kernel: one thread per (token, expert), 4096-long serial
// dot. pp16384 profile: 26.9 s across 58 layers, far below FMA peak -- probe
// latency, LDS reuse and streaming variants.
//
// tools/bench/build.sh nem_router_bench && /tmp/nem_router_bench [M...]

#include <hip/hip_runtime.h>

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

constexpr int kN = 512;   // experts
constexpr int kK = 4096;  // hidden

// A: production baseline (one thread per (token, expert) output).
__global__ void GemmNaive(const float* __restrict__ x,
                          const float* __restrict__ w,
                          float* __restrict__ out, int n, int k, int m) {
  const int t = blockIdx.y;
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  const float* xr = x + static_cast<int64_t>(t) * k;
  const float* wr = w + static_cast<int64_t>(r) * k;
  float acc = 0.0F;
  for (int i = 0; i < k; ++i) acc += xr[i] * wr[i];
  out[static_cast<int64_t>(t) * n + r] = acc;
}

// B: k-split 128 lanes + intra-block tree reduction, one block per token.
// Thread owns a 32-element k slice and streams all 512 expert rows in it.
__global__ void GemmKsplit(const float* __restrict__ x,
                           const float* __restrict__ w,
                           float* __restrict__ out, int n, int k, int m) {
  const int t = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);  // 128
  const int j = tid;                              // lane in k-split
  const float* xr = x + static_cast<int64_t>(t) * k;
  __shared__ float red[4];
  for (int r = 0; r < n; ++r) {
    const float* wr = w + static_cast<int64_t>(r) * k;
    float acc = 0.0F;
#pragma unroll
    for (int e = 0; e < 32; ++e) acc += xr[j * 32 + e] * wr[j * 32 + e];
    // warp reduce
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1)
      acc += __shfl_down_sync(0xFFFFFFFFull, acc, s);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
      out[static_cast<int64_t>(t) * n + r] = red[0] + red[1] + red[2] +
                                             red[3];
    }
    __syncthreads();
  }
}

// A2: production mapping + float4 loads + 4 independent accumulators.
__global__ void GemmNaive4(const float* __restrict__ x,
                           const float* __restrict__ w,
                           float* __restrict__ out, int n, int k, int m) {
  const int t = blockIdx.y;
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  const float4* xv = reinterpret_cast<const float4*>(
      x + static_cast<int64_t>(t) * k);
  const float4* wv = reinterpret_cast<const float4*>(
      w + static_cast<int64_t>(r) * k);
  float a0 = 0.0F, a1 = 0.0F, a2 = 0.0F, a3 = 0.0F;
#pragma unroll 1
  for (int i = 0; i < k / 4; ++i) {
    const float4 a = xv[i];
    const float4 b = wv[i];
    a0 += a.x * b.x;
    a1 += a.y * b.y;
    a2 += a.z * b.z;
    a3 += a.w * b.w;
  }
  out[static_cast<int64_t>(t) * n + r] = a0 + a1 + (a2 + a3);
}

// C: canonical tiled SGEMM. 256 threads, thread grid (256/TGR) x TGR, tile
// TM(tokens) x TN(experts), k-step KS, float4 global loads, LDS stage with
// +1 pad, register accumulation. Each thread owns TT x TR outputs.
template <int TM, int TN, int KS, int TGR>
__global__ void GemmTile(const float* __restrict__ x,
                         const float* __restrict__ w,
                         float* __restrict__ out, int n, int k, int m) {
  constexpr int TGT = 256 / TGR;
  constexpr int TT = TM / TGT;           // tokens per thread
  constexpr int TR = TN / TGR;           // experts per thread
  const int t0 = blockIdx.x * TM;
  const int r0 = blockIdx.y * TN;
  const int tid = static_cast<int>(threadIdx.x);  // 256
  const int ty = tid / TGR;
  const int tx = tid % TGR;
  __shared__ float xs[TM][KS + 1];
  __shared__ float ws[TN][KS + 1];
  const float4* xp = reinterpret_cast<const float4*>(x +
      static_cast<int64_t>(t0) * k);
  const float4* wp = reinterpret_cast<const float4*>(w +
      static_cast<int64_t>(r0) * k);
  float acc[TT][TR];
#pragma unroll 1
  for (int i = 0; i < TT; ++i)
#pragma unroll 1
    for (int j = 0; j < TR; ++j) acc[i][j] = 0.0F;
  for (int kc = 0; kc < k; kc += KS) {
    for (int i = tid; i < TM * (KS / 4); i += 256) {
      const int row = i / (KS / 4);
      const int c4 = i % (KS / 4);
      float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      if (t0 + row < m)
        v = *reinterpret_cast<const float4*>(xp +
            static_cast<int64_t>(row) * (k / 4) + (kc / 4) + c4);
      xs[row][c4 * 4 + 0] = v.x;
      xs[row][c4 * 4 + 1] = v.y;
      xs[row][c4 * 4 + 2] = v.z;
      xs[row][c4 * 4 + 3] = v.w;
    }
    for (int i = tid; i < TN * (KS / 4); i += 256) {
      const int row = i / (KS / 4);
      const int c4 = i % (KS / 4);
      const float4 v = *reinterpret_cast<const float4*>(wp +
          static_cast<int64_t>(row) * (k / 4) + (kc / 4) + c4);
      ws[row][c4 * 4 + 0] = v.x;
      ws[row][c4 * 4 + 1] = v.y;
      ws[row][c4 * 4 + 2] = v.z;
      ws[row][c4 * 4 + 3] = v.w;
    }
    __syncthreads();
#pragma unroll 1
    for (int e = 0; e < KS; ++e) {
      float ra[TT], rb[TR];
#pragma unroll
      for (int i = 0; i < TT; ++i) ra[i] = xs[ty * TT + i][e];
#pragma unroll
      for (int j = 0; j < TR; ++j) rb[j] = ws[tx * TR + j][e];
#pragma unroll
      for (int i = 0; i < TT; ++i)
#pragma unroll
        for (int j = 0; j < TR; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
#pragma unroll 1
  for (int i = 0; i < TT; ++i) {
    const int t = t0 + ty * TT + i;
    if (t >= m) break;
#pragma unroll
    for (int j = 0; j < TR; ++j)
      out[static_cast<int64_t>(t) * n + r0 + tx * TR + j] = acc[i][j];
  }
}

template <int TM, int TN, int KS, int TGR>
void LaunchTile(const float* x, const float* w, float* out, int m,
                hipStream_t s) {
  static_assert(TM % (256 / TGR) == 0 && TN % TGR == 0, "bad tile split");
  const dim3 grid((m + TM - 1) / TM, kN / TN);
  GemmTile<TM, TN, KS, TGR><<<grid, 256, 0, s>>>(x, w, out, kN, kK, m);
}

// G: tiled SGEMM with register prefetch of the next k-index across LDS.
template <int TM, int TN, int KS, int TGR>
__global__ void GemmTileP(const float* __restrict__ x,
                          const float* __restrict__ w,
                          float* __restrict__ out, int n, int k, int m) {
  constexpr int TGT = 256 / TGR;
  constexpr int TT = TM / TGT;
  constexpr int TR = TN / TGR;
  const int t0 = blockIdx.x * TM;
  const int r0 = blockIdx.y * TN;
  const int tid = static_cast<int>(threadIdx.x);
  const int ty = tid / TGR;
  const int tx = tid % TGR;
  __shared__ float xs[TM][KS + 1];
  __shared__ float ws[TN][KS + 1];
  const float4* xp = reinterpret_cast<const float4*>(x +
      static_cast<int64_t>(t0) * k);
  const float4* wp = reinterpret_cast<const float4*>(w +
      static_cast<int64_t>(r0) * k);
  float acc[TT][TR];
#pragma unroll 1
  for (int i = 0; i < TT; ++i)
#pragma unroll 1
    for (int j = 0; j < TR; ++j) acc[i][j] = 0.0F;
  for (int kc = 0; kc < k; kc += KS) {
    for (int i = tid; i < TM * (KS / 4); i += 256) {
      const int row = i / (KS / 4);
      const int c4 = i % (KS / 4);
      float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      if (t0 + row < m)
        v = *reinterpret_cast<const float4*>(xp +
            static_cast<int64_t>(row) * (k / 4) + (kc / 4) + c4);
      xs[row][c4 * 4 + 0] = v.x;
      xs[row][c4 * 4 + 1] = v.y;
      xs[row][c4 * 4 + 2] = v.z;
      xs[row][c4 * 4 + 3] = v.w;
    }
    for (int i = tid; i < TN * (KS / 4); i += 256) {
      const int row = i / (KS / 4);
      const int c4 = i % (KS / 4);
      const float4 v = *reinterpret_cast<const float4*>(wp +
          static_cast<int64_t>(row) * (k / 4) + (kc / 4) + c4);
      ws[row][c4 * 4 + 0] = v.x;
      ws[row][c4 * 4 + 1] = v.y;
      ws[row][c4 * 4 + 2] = v.z;
      ws[row][c4 * 4 + 3] = v.w;
    }
    __syncthreads();
    float ra[TT], rb[TR], ra2[TT], rb2[TR];
#pragma unroll
    for (int i = 0; i < TT; ++i) ra[i] = xs[ty * TT + i][0];
#pragma unroll
    for (int j = 0; j < TR; ++j) rb[j] = ws[tx * TR + j][0];
#pragma unroll 1
    for (int e = 0; e < KS; ++e) {
      if (e + 1 < KS) {
#pragma unroll
        for (int i = 0; i < TT; ++i) ra2[i] = xs[ty * TT + i][e + 1];
#pragma unroll
        for (int j = 0; j < TR; ++j) rb2[j] = ws[tx * TR + j][e + 1];
      }
#pragma unroll
      for (int i = 0; i < TT; ++i)
#pragma unroll
        for (int j = 0; j < TR; ++j) acc[i][j] += ra[i] * rb[j];
#pragma unroll
      for (int i = 0; i < TT; ++i) ra[i] = ra2[i];
#pragma unroll
      for (int j = 0; j < TR; ++j) rb[j] = rb2[j];
    }
    __syncthreads();
  }
#pragma unroll 1
  for (int i = 0; i < TT; ++i) {
    const int t = t0 + ty * TT + i;
    if (t >= m) break;
#pragma unroll
    for (int j = 0; j < TR; ++j)
      out[static_cast<int64_t>(t) * n + r0 + tx * TR + j] = acc[i][j];
  }
}

template <int TM, int TN, int KS, int TGR>
void LaunchTileP(const float* x, const float* w, float* out, int m,
                 hipStream_t s) {
  const dim3 grid((m + TM - 1) / TM, kN / TN);
  GemmTileP<TM, TN, KS, TGR><<<grid, 256, 0, s>>>(x, w, out, kN, kK, m);
}

void LaunchNaive(const float* x, const float* w, float* out, int m,
                hipStream_t s) {
  const dim3 grid((kN + 255) / 256, m);
  GemmNaive<<<grid, 256, 0, s>>>(x, w, out, kN, kK, m);
}

void LaunchKsplit(const float* x, const float* w, float* out, int m,
                  hipStream_t s) {
  GemmKsplit<<<m, 128, 0, s>>>(x, w, out, kN, kK, m);
}

void LaunchNaive4v(const float* x, const float* w, float* out, int m,
                   hipStream_t s) {
  const dim3 grid((kN + 255) / 256, m);
  GemmNaive4<<<grid, 256, 0, s>>>(x, w, out, kN, kK, m);
}

void LaunchTileC(const float* x, const float* w, float* out, int m,
                 hipStream_t s) { LaunchTile<64, 64, 16, 16>(x, w, out, m, s); }
void LaunchTileC32(const float* x, const float* w, float* out, int m,
                   hipStream_t s) { LaunchTile<64, 64, 32, 16>(x, w, out, m, s); }
void LaunchTileE(const float* x, const float* w, float* out, int m,
                 hipStream_t s) { LaunchTile<64, 32, 16, 8>(x, w, out, m, s); }
void LaunchTileE32(const float* x, const float* w, float* out, int m,
                   hipStream_t s) { LaunchTile<64, 32, 32, 8>(x, w, out, m, s); }
void LaunchTileF(const float* x, const float* w, float* out, int m,
                 hipStream_t s) { LaunchTile<64, 64, 64, 16>(x, w, out, m, s); }
void LaunchTileG(const float* x, const float* w, float* out, int m,
                 hipStream_t s) { LaunchTileP<64, 64, 32, 16>(x, w, out, m, s); }
void LaunchTileH(const float* x, const float* w, float* out, int m,
                 hipStream_t s) { LaunchTileP<64, 64, 64, 16>(x, w, out, m, s); }
void LaunchTileI(const float* x, const float* w, float* out, int m,
                 hipStream_t s) { LaunchTile<64, 64, 32, 32>(x, w, out, m, s); }
void LaunchTileJ(const float* x, const float* w, float* out, int m,
                 hipStream_t s) { LaunchTile<128, 64, 32, 16>(x, w, out, m, s); }
void LaunchTileK(const float* x, const float* w, float* out, int m,
                 hipStream_t s) { LaunchTile<128, 64, 32, 32>(x, w, out, m, s); }
void LaunchTileI16(const float* x, const float* w, float* out, int m,
                   hipStream_t s) { LaunchTile<64, 64, 16, 32>(x, w, out, m, s); }

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> lens;
  if (argc > 1) {
    for (int i = 1; i < argc; ++i) lens.push_back(atoi(argv[i]));
  } else {
    lens = {2048, 8192, 16384};
  }
  int maxlen = 0;
  for (int L : lens) maxlen = L > maxlen ? L : maxlen;

  std::vector<float> hx(static_cast<size_t>(maxlen) * kK);
  std::vector<float> hw(static_cast<size_t>(kN) * kK);
  uint32_t st = 987654321U;
  auto rnd = [&st]() {
    st = st * 1664525U + 1013904223U;
    return static_cast<float>(static_cast<int>(st >> 8) % 20000 - 10000) /
           10000.0F;
  };
  for (auto& v : hx) v = rnd() * 0.001F;
  for (auto& v : hw) v = rnd() * 0.001F;

  float *dx, *dw, *dout;
  CHECK(hipMalloc(&dx, hx.size() * sizeof(float)));
  CHECK(hipMalloc(&dw, hw.size() * sizeof(float)));
  CHECK(hipMalloc(&dout, static_cast<size_t>(maxlen) * kN * sizeof(float)));
  CHECK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dw, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));

  // f64 reference at 64 tokens.
  const int clen = 64;
  std::vector<double> ref(static_cast<size_t>(clen) * kN);
  for (int t = 0; t < clen; ++t) {
    for (int r = 0; r < kN; ++r) {
      double acc = 0.0;
      for (int i = 0; i < kK; ++i)
        acc += static_cast<double>(hx[static_cast<size_t>(t) * kK + i]) *
               static_cast<double>(hw[static_cast<size_t>(r) * kK + i]);
      ref[static_cast<size_t>(t) * kN + r] = acc;
    }
  }
  struct V {
    const char* name;
    void (*run)(const float*, const float*, float*, int, hipStream_t);
  };
  const V variants[] = {
      {"A naive prod       ", LaunchNaive},
      {"A2 naive4 float4v  ", LaunchNaive4v},
      {"C  64x64 ks16 g16  ", LaunchTileC},
      {"C32 64x64 ks32 g16 ", LaunchTileC32},
      {"E  64x32 ks16 g8   ", LaunchTileE},
      {"E32 64x32 ks32 g8  ", LaunchTileE32},
      {"F  64x64 ks64 g16  ", LaunchTileF},
      {"G  64x64p ks32 g16 ", LaunchTileG},
      {"H  64x64p ks64 g16 ", LaunchTileH},
      {"I  64x64 ks32 g32  ", LaunchTileI},
      {"J 128x64 ks32 g16  ", LaunchTileJ},
      {"K 128x64 ks32 g32  ", LaunchTileK},
      {"I16 64x64 ks16 g32 ", LaunchTileI16},
  };
  std::vector<float> cand(static_cast<size_t>(clen) * kN);
  std::printf("correctness (first %d tokens):\n", clen);
  for (const auto& v : variants) {
    CHECK(hipMemset(dout, 0, static_cast<size_t>(clen) * kN * 4));
    v.run(dx, dw, dout, clen, nullptr);
    CHECK(hipDeviceSynchronize());
    if (hipGetLastError() != hipSuccess) {
      std::printf("  %s LAUNCH FAILED\n", v.name);
      return 1;
    }
    CHECK(hipMemcpy(cand.data(), dout, cand.size() * 4,
                    hipMemcpyDeviceToHost));
    double num = 0.0, den = 0.0;
    size_t nan = 0;
    for (size_t i = 0; i < cand.size(); ++i) {
      if (std::isnan(cand[i])) ++nan;
      const double d = static_cast<double>(cand[i]) - ref[i];
      num += d * d;
      den += ref[i] * ref[i];
    }
    const double rel = std::sqrt(num / den);
    std::printf("  %s rel=%.3e nan=%zu %s\n", v.name, rel, nan,
                (nan || rel > 5e-5) ? "FAIL" : "OK");
    if (nan || rel > 5e-5) return 1;
  }

  std::printf("\n%-20s", "variant");
  for (size_t i = 0; i < lens.size(); ++i) std::printf("%12s", "TF/s");
  std::printf("   (M)\n");
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  for (const auto& v : variants) {
    std::printf("%-20s", v.name);
    for (int L : lens) {
      for (int i = 0; i < 2; ++i) v.run(dx, dw, dout, L, nullptr);
      CHECK(hipDeviceSynchronize());
      CHECK(hipEventRecord(e0));
      const int reps = L >= 8192 ? 3 : 10;
      for (int i = 0; i < reps; ++i) v.run(dx, dw, dout, L, nullptr);
      CHECK(hipEventRecord(e1));
      CHECK(hipEventSynchronize(e1));
      float ms = 0.0F;
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      const double mops = 2.0 * L * kN * kK * reps;
      std::printf("%12.1f", mops / (ms * 1e-3) / 1e12);
    }
    std::printf("\n");
  }
  return 0;
}
