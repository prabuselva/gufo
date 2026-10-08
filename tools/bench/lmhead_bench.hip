// Q8_0 GEMV variants across the real qwen36 decode shapes. Measures achieved
// GB/s for the current byte-per-lane kernel vs bit-exact-preserving scheduling
// variants (unroll depth, rows-per-block) and a non-exact vectorized-load
// ceiling probe. Each bit-exact variant is checked against the current kernel
// for identical output bits; the vec probe is reported as a bandwidth ceiling
// only (it changes the accumulation order and is not promotable).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "hip err %s @%d\n", hipGetErrorString(e), __LINE__); \
  exit(1); } } while (0)

struct Q8_0Block { __half d; std::int8_t qs[32]; };

__device__ __forceinline__ float WSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
  return v;
}

// Current production kernel: 4 rows/block (128 threads), byte per lane per
// block, unroll 4. This is the bit-exact reference.
__global__ void kCurrent(const Q8_0Block* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ out,
                         std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) return;
  const std::uint32_t nb = cols / 32;
  const Q8_0Block* __restrict__ row = w + (std::size_t)r * nb;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nb; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * (float)row[b].qs[lane] * x[b * 32 + lane];
  }
  acc = WSum(acc);
  if (lane == 0) out[r] = acc;
}

// Bit-exact: same per-row FMA sequence, deeper unroll (more loads in flight).
__global__ void kUnroll8(const Q8_0Block* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ out,
                         std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) return;
  const std::uint32_t nb = cols / 32;
  const Q8_0Block* __restrict__ row = w + (std::size_t)r * nb;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 8
  for (std::uint32_t b = 0; b < nb; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * (float)row[b].qs[lane] * x[b * 32 + lane];
  }
  acc = WSum(acc);
  if (lane == 0) out[r] = acc;
}

// Bit-exact: 8 rows/block (256 threads), same per-row FMA sequence.
__global__ void kRows8(const Q8_0Block* __restrict__ w,
                       const float* __restrict__ x, float* __restrict__ out,
                       std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 8U + (threadIdx.x >> 5);
  if (r >= rows) return;
  const std::uint32_t nb = cols / 32;
  const Q8_0Block* __restrict__ row = w + (std::size_t)r * nb;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nb; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * (float)row[b].qs[lane] * x[b * 32 + lane];
  }
  acc = WSum(acc);
  if (lane == 0) out[r] = acc;
}

// Bit-exact: 2 rows/block (64 threads), same per-row FMA sequence.
__global__ void kRows2(const Q8_0Block* __restrict__ w,
                       const float* __restrict__ x, float* __restrict__ out,
                       std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 2U + (threadIdx.x >> 5);
  if (r >= rows) return;
  const std::uint32_t nb = cols / 32;
  const Q8_0Block* __restrict__ row = w + (std::size_t)r * nb;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nb; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * (float)row[b].qs[lane] * x[b * 32 + lane];
  }
  acc = WSum(acc);
  if (lane == 0) out[r] = acc;
}

// NOT bit-exact (ceiling probe): uint32 quant load + float4 x. Reads the row as
// a flat byte stream, so the products are grouped differently.
__global__ void kVec4(const Q8_0Block* __restrict__ w,
                      const float* __restrict__ x, float* __restrict__ out,
                      std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) return;
  const std::uint32_t nb = cols / 32;
  const std::uint8_t* __restrict__ row =
      reinterpret_cast<const std::uint8_t*>(w) + (std::size_t)r * nb * 34;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nb; ++b) {
    const float d = __half2float(*reinterpret_cast<const __half*>(row + b * 34));
    const std::uint32_t q = *reinterpret_cast<const std::uint32_t*>(
        row + b * 34 + 2 + lane * 4);
    const float4 xv = *reinterpret_cast<const float4*>(x + b * 32 + lane * 4);
    acc += d * ((float)(std::int8_t)(q & 0xFF) * xv.x +
                (float)(std::int8_t)((q >> 8) & 0xFF) * xv.y +
                (float)(std::int8_t)((q >> 16) & 0xFF) * xv.z +
                (float)(std::int8_t)(q >> 24) * xv.w);
  }
  acc = WSum(acc);
  if (lane == 0) out[r] = acc;
}

typedef void (*Launch)(const Q8_0Block*, const float*, float*, std::uint32_t,
                       std::uint32_t, hipStream_t);

static void lCurrent(const Q8_0Block* w, const float* x, float* o,
                     std::uint32_t r, std::uint32_t c, hipStream_t s) {
  kCurrent<<<(r + 3) / 4, 128, 0, s>>>(w, x, o, r, c);
}
static void lUnroll8(const Q8_0Block* w, const float* x, float* o,
                     std::uint32_t r, std::uint32_t c, hipStream_t s) {
  kUnroll8<<<(r + 3) / 4, 128, 0, s>>>(w, x, o, r, c);
}
static void lRows8(const Q8_0Block* w, const float* x, float* o, std::uint32_t r,
                   std::uint32_t c, hipStream_t s) {
  kRows8<<<(r + 7) / 8, 256, 0, s>>>(w, x, o, r, c);
}
static void lRows2(const Q8_0Block* w, const float* x, float* o, std::uint32_t r,
                   std::uint32_t c, hipStream_t s) {
  kRows2<<<(r + 1) / 2, 64, 0, s>>>(w, x, o, r, c);
}
static void lVec4(const Q8_0Block* w, const float* x, float* o, std::uint32_t r,
                  std::uint32_t c, hipStream_t s) {
  kVec4<<<(r + 3) / 4, 128, 0, s>>>(w, x, o, r, c);
}

int main() {
  struct Shape { const char* name; std::uint32_t rows; std::uint32_t cols; };
  const Shape shapes[] = {
      {"lm_head  248320x2048", 248320, 2048},
      {"ssm_qkv    8192x2048", 8192, 2048},
      {"ssm_gate   4096x2048", 4096, 2048},
      {"attn_q     4096x2048", 4096, 2048},
      {"moe_gate    512x2048", 512, 2048},
  };
  const struct { const char* name; Launch fn; bool exact; } kernels[] = {
      {"current", lCurrent, true},
      {"unroll8", lUnroll8, true},
      {"rows8", lRows8, true},
      {"rows2", lRows2, true},
      {"vec4*", lVec4, false},
  };

  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  const int iters = 30;

  for (const Shape& sh : shapes) {
    const std::size_t wbytes = (std::size_t)sh.rows * (sh.cols / 32) * 34;
    const double gb = wbytes / 1e9;
    float *x, *out, *ref;
    Q8_0Block* w;
    CK(hipMalloc(&w, wbytes));
    CK(hipMalloc(&x, sh.cols * sizeof(float)));
    CK(hipMalloc(&out, sh.rows * sizeof(float)));
    CK(hipMalloc(&ref, sh.rows * sizeof(float)));
    // Fill weights with a nonzero pattern (scales land on even offsets).
    CK(hipMemset(w, 0x37, wbytes));
    CK(hipMemset(x, 0x3c, sh.cols * sizeof(float)));
    printf("\n== %s  (%.1f MB weights) ==\n", sh.name, gb * 1000);
    for (const auto& k : kernels) {
      // Warm.
      k.fn(w, x, out, sh.rows, sh.cols, 0);
      CK(hipDeviceSynchronize());
      hipEventRecord(e0);
      for (int i = 0; i < iters; ++i) k.fn(w, x, out, sh.rows, sh.cols, 0);
      hipEventRecord(e1);
      CK(hipDeviceSynchronize());
      float ms;
      hipEventElapsedTime(&ms, e0, e1);
      ms /= iters;
      const char* exact = "?";
      if (k.exact) {
        // Re-run current into ref, compare bits.
        lCurrent(w, x, ref, sh.rows, sh.cols, 0);
        CK(hipDeviceSynchronize());
        std::vector<float> a(sh.rows), b(sh.rows);
        CK(hipMemcpy(a.data(), out, sh.rows * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), ref, sh.rows * 4, hipMemcpyDeviceToHost));
        bool same = true;
        for (std::uint32_t i = 0; i < sh.rows; ++i) {
          if (std::memcmp(&a[i], &b[i], 4) != 0) { same = false; break; }
        }
        exact = same ? "exact" : "DIFF!";
      }
      printf("  %-9s %8.3f ms  %6.1f GB/s  %s\n", k.name, ms,
             gb / (ms / 1e3), exact);
    }
    CK(hipFree(w));
    CK(hipFree(x));
    CK(hipFree(out));
    CK(hipFree(ref));
  }
  return 0;
}