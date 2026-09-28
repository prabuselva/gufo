// lm_head GEMV variants: rows=248320, cols=2048, Q8_0. Measures achieved GB/s
// for the current byte-per-lane kernel vs vectorized unpacks.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "hip err %s @%d\n", hipGetErrorString(e), __LINE__); \
  exit(1); } } while (0)

struct Q8_0Block { __half d; std::int8_t qs[32]; };

__device__ __forceinline__ float WSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
  return v;
}

// Current production kernel: 4 rows/block, byte per lane per block.
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

// Variant A: uint32 quant load (misaligned on even blocks) + float4 x.
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

// Variant B: two blocks per iteration, uint16 quant load (always 2B aligned)
// + float2 x (always 8B aligned).
__global__ void kVec2(const Q8_0Block* __restrict__ w,
                      const float* __restrict__ x, float* __restrict__ out,
                      std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) return;
  const std::uint32_t nb = cols / 32;
  const std::uint8_t* __restrict__ row =
      reinterpret_cast<const std::uint8_t*>(w) + (std::size_t)r * nb * 34;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 2
  for (std::uint32_t b = 0; b < nb; b += 2) {
    const float d0 = __half2float(*reinterpret_cast<const __half*>(row + b * 34));
    const float d1 =
        __half2float(*reinterpret_cast<const __half*>(row + b * 34 + 34));
    const std::uint16_t q = *reinterpret_cast<const std::uint16_t*>(
        row + b * 34 + 2 + lane * 2);
    const float2 xv = *reinterpret_cast<const float2*>(x + b * 32 + lane * 2);
    const float2 xv2 = *reinterpret_cast<const float2*>(x + b * 32 + 32 + lane * 2);
    const std::uint16_t q2 = *reinterpret_cast<const std::uint16_t*>(
        row + b * 34 + 34 + 2 + lane * 2);
    acc += d0 * ((float)(std::int8_t)(q & 0xFF) * xv.x +
                 (float)(std::int8_t)(q >> 8) * xv.y);
    acc += d1 * ((float)(std::int8_t)(q2 & 0xFF) * xv2.x +
                 (float)(std::int8_t)(q2 >> 8) * xv2.y);
  }
  acc = WSum(acc);
  if (lane == 0) out[r] = acc;
}

// Variant C: vec4 with x staged in LDS once per block.
__global__ void kVec4Lds(const Q8_0Block* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ out,
                         std::uint32_t rows, std::uint32_t cols) {
  extern __shared__ float sx[];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = x[i];
  __syncthreads();
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
    const float4 xv = *reinterpret_cast<const float4*>(sx + b * 32 + lane * 4);
    acc += d * ((float)(std::int8_t)(q & 0xFF) * xv.x +
                (float)(std::int8_t)((q >> 8) & 0xFF) * xv.y +
                (float)(std::int8_t)((q >> 16) & 0xFF) * xv.z +
                (float)(std::int8_t)(q >> 24) * xv.w);
  }
  acc = WSum(acc);
  if (lane == 0) out[r] = acc;
}

int main() {
  const std::uint32_t rows = 248320, cols = 2048;
  const std::size_t wbytes = (std::size_t)rows * (cols / 32) * 34;
  float *x, *out; Q8_0Block* w;
  CK(hipMalloc(&w, wbytes));
  CK(hipMalloc(&x, cols * sizeof(float)));
  CK(hipMalloc(&out, rows * sizeof(float)));
  CK(hipMemset(w, 0x37, wbytes));
  CK(hipMemset(x, 0x3c, cols * sizeof(float)));
  const double gb = wbytes / 1e9;
  printf("weights %.1f MB\n", gb * 1000);

  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  const int iters = 20;
  auto bench = [&](const char* name, void (*launch)(const Q8_0Block*, const float*, float*, std::uint32_t, std::uint32_t, hipStream_t)) {
    launch(w, x, out, rows, cols, 0);
    CK(hipDeviceSynchronize());
    hipEventRecord(e0);
    for (int i = 0; i < iters; ++i) launch(w, x, out, rows, cols, 0);
    hipEventRecord(e1);
    CK(hipDeviceSynchronize());
    float ms; hipEventElapsedTime(&ms, e0, e1);
    ms /= iters;
    printf("%-12s %7.3f ms  %6.1f GB/s\n", name, ms, gb / (ms / 1e3));
  };
  bench("current", [](const Q8_0Block* w, const float* x, float* o, std::uint32_t r, std::uint32_t c, hipStream_t s) {
    kCurrent<<<(r + 3) / 4, 128, 0, s>>>(w, x, o, r, c); });
  bench("vec4", [](const Q8_0Block* w, const float* x, float* o, std::uint32_t r, std::uint32_t c, hipStream_t s) {
    kVec4<<<(r + 3) / 4, 128, 0, s>>>(w, x, o, r, c); });
  bench("vec2", [](const Q8_0Block* w, const float* x, float* o, std::uint32_t r, std::uint32_t c, hipStream_t s) {
    kVec2<<<(r + 3) / 4, 128, 0, s>>>(w, x, o, r, c); });
  bench("vec4lds", [](const Q8_0Block* w, const float* x, float* o, std::uint32_t r, std::uint32_t c, hipStream_t s) {
    kVec4Lds<<<(r + 3) / 4, 128, 2048 * 4, s>>>(w, x, o, r, c); });
  return 0;
}