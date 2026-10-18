// Q8_0 GEMV probe on the Ling3 KDA attention K/V shape (rows=4096,
// cols=2560) and shared-expert shape (rows=768, cols=2560). Production
// kernel is V0 (GemvQ8_0Vec4 copy); V1 pairs two rows per warp sharing the
// x loads; V2 pairs rows with 256-thread blocks; V3 drops scale loads as a
// diagnostic ceiling. V0/V1/V2 must agree bit-exactly.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                             \
  do {                                                                         \
    hipError_t err = (cmd);                                                    \
    if (err != hipSuccess) {                                                   \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err),  \
                   __FILE__, __LINE__);                                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct alignas(2) Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "Q8_0 block layout must stay 34B");

__device__ __forceinline__ float WarpReduceSum(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xFFFFFFFFull, v, off);
  }
  return v;
}

__global__ void V0(const Q8_0Block* __restrict__ w, const float* __restrict__ x,
                   float* __restrict__ out, std::uint32_t rows,
                   std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t groups = cols / 128U;
  const Q8_0Block* __restrict__ row = w + static_cast<std::size_t>(r) * (cols / 32U);
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t g = 0; g < groups; ++g) {
    const std::uint32_t blk = g * 4U + lane / 8U;
    const std::uint32_t word = (4U * lane) % 32U;
    std::int32_t packed;
    __builtin_memcpy(&packed, &row[blk].qs[word], 4);
    const float d = __half2float(row[blk].d);
    const std::uint32_t k = blk * 32U + word;
    acc += d * static_cast<float>(static_cast<std::int8_t>(packed & 0xFF)) * x[k];
    acc += d * static_cast<float>(static_cast<std::int8_t>((packed >> 8) & 0xFF)) * x[k + 1];
    acc += d * static_cast<float>(static_cast<std::int8_t>((packed >> 16) & 0xFF)) * x[k + 2];
    acc += d * static_cast<float>(static_cast<std::int8_t>((packed >> 24) & 0xFF)) * x[k + 3];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

// Two rows per warp; the same four x values feed both rows. Per-row sum
// order matches V0 exactly, so results are bit-identical. WARPS = rows per
// block / 4.
template <int WARPS>
__global__ void V1(const Q8_0Block* __restrict__ w, const float* __restrict__ x,
                   float* __restrict__ out, std::uint32_t rows,
                   std::uint32_t cols) {
  const std::uint32_t pair = blockIdx.x * WARPS + (threadIdx.x >> 5);
  const std::uint32_t r0 = pair * 2U;
  if (r0 >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 32U;
  const std::uint32_t groups = cols / 128U;
  const bool two = r0 + 1U < rows;
  const Q8_0Block* __restrict__ row0 = w + static_cast<std::size_t>(r0) * nblocks;
  const Q8_0Block* __restrict__ row1 = two ? row0 + nblocks : row0;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
#pragma unroll 4
  for (std::uint32_t g = 0; g < groups; ++g) {
    std::uint32_t k = 0;
    float x0 = 0, x1 = 0, x2 = 0, x3 = 0, d0 = 0, d1 = 0;
    std::int32_t p0 = 0, p1 = 0;
#pragma unroll
    for (int side = 0; side < (two ? 2 : 1); ++side) {
      const Q8_0Block* row = side ? row1 : row0;
      const std::uint32_t blk = g * 4U + lane / 8U;
      const std::uint32_t word = (4U * lane) % 32U;
      std::int32_t packed;
      __builtin_memcpy(&packed, &row[blk].qs[word], 4);
      const float d = __half2float(row[blk].d);
      k = blk * 32U + word;
      if (side) {
        p1 = packed;
        d1 = d;
      } else {
        p0 = packed;
        d0 = d;
        x0 = x[k]; x1 = x[k + 1]; x2 = x[k + 2]; x3 = x[k + 3];
      }
    }
    acc0 += d0 * static_cast<float>(static_cast<std::int8_t>(p0 & 0xFF)) * x0;
    acc0 += d0 * static_cast<float>(static_cast<std::int8_t>((p0 >> 8) & 0xFF)) * x1;
    acc0 += d0 * static_cast<float>(static_cast<std::int8_t>((p0 >> 16) & 0xFF)) * x2;
    acc0 += d0 * static_cast<float>(static_cast<std::int8_t>((p0 >> 24) & 0xFF)) * x3;
    if (two) {
      acc1 += d1 * static_cast<float>(static_cast<std::int8_t>(p1 & 0xFF)) * x0;
      acc1 += d1 * static_cast<float>(static_cast<std::int8_t>((p1 >> 8) & 0xFF)) * x1;
      acc1 += d1 * static_cast<float>(static_cast<std::int8_t>((p1 >> 16) & 0xFF)) * x2;
      acc1 += d1 * static_cast<float>(static_cast<std::int8_t>((p1 >> 24) & 0xFF)) * x3;
    }
  }
  acc0 = WarpReduceSum(acc0);
  acc1 = WarpReduceSum(acc1);
  if (lane == 0) {
    out[r0] = acc0;
    if (two) {
      out[r0 + 1U] = acc1;
    }
  }
}

// Scale-load diagnostic: same walk, d replaced by 1.0f (timing only).
__global__ void V3(const Q8_0Block* __restrict__ w, const float* __restrict__ x,
                   float* __restrict__ out, std::uint32_t rows,
                   std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t groups = cols / 128U;
  const Q8_0Block* __restrict__ row = w + static_cast<std::size_t>(r) * (cols / 32U);
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t g = 0; g < groups; ++g) {
    const std::uint32_t blk = g * 4U + lane / 8U;
    const std::uint32_t word = (4U * lane) % 32U;
    std::int32_t packed;
    __builtin_memcpy(&packed, &row[blk].qs[word], 4);
    const std::uint32_t k = blk * 32U + word;
    acc += 1.0f * static_cast<float>(static_cast<std::int8_t>(packed & 0xFF)) * x[k];
    acc += 1.0f * static_cast<float>(static_cast<std::int8_t>((packed >> 8) & 0xFF)) * x[k + 1];
    acc += 1.0f * static_cast<float>(static_cast<std::int8_t>((packed >> 16) & 0xFF)) * x[k + 2];
    acc += 1.0f * static_cast<float>(static_cast<std::int8_t>((packed >> 24) & 0xFF)) * x[k + 3];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

using Kern = void (*)(const Q8_0Block*, const float*, float*, std::uint32_t, std::uint32_t);

static float Time(Kern k, int wblocks, int threads, const Q8_0Block* w,
                  const float* x, float* out, std::uint32_t rows,
                  std::uint32_t cols, int reps) {
  for (int i = 0; i < 20; ++i) {
    k<<<wblocks, threads>>>(w, x, out, rows, cols);
  }
  CHECK(hipDeviceSynchronize());
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  float best = 1e30f;
  for (int trial = 0; trial < 5; ++trial) {
    CHECK(hipEventRecord(a));
    for (int i = 0; i < reps; ++i) {
      k<<<wblocks, threads>>>(w, x, out, rows, cols);
    }
    CHECK(hipEventRecord(b));
    CHECK(hipEventSynchronize(b));
    float ms = 0.0f;
    CHECK(hipEventElapsedTime(&ms, a, b));
    if (ms < best) best = ms;
  }
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
  return best * 1000.0f / reps;  // us
}

int main() {
  CHECK(hipSetDevice(0));
  const std::uint32_t cols = 2560U;
  const std::size_t nblocks = cols / 32U;
  float* x = nullptr;
  CHECK(hipMalloc(&x, cols * sizeof(float)));
  {
    float* hx = new float[cols];
    for (std::uint32_t i = 0; i < cols; ++i) hx[i] = ((i % 97) - 48) * 0.125f;
    CHECK(hipMemcpy(x, hx, cols * sizeof(float), hipMemcpyHostToDevice));
    delete[] hx;
  }
  struct Shape { std::uint32_t rows; const char* name; };
  for (Shape sh : {Shape{4096, "kda_kv"}, Shape{768, "shexp"},
                   Shape{100000, "hbm_272MB"}}) {
    const std::uint32_t rows = sh.rows;
    const std::size_t wbytes = (std::size_t)rows * nblocks * sizeof(Q8_0Block);
    Q8_0Block* w = nullptr;
    float *o0 = nullptr, *o1 = nullptr;
    CHECK(hipMalloc(&w, wbytes));
    CHECK(hipMalloc(&o0, rows * sizeof(float)));
    CHECK(hipMalloc(&o1, rows * sizeof(float)));
    {
      unsigned char* hw = new unsigned char[wbytes];
      unsigned state = 12345u + rows;
      for (std::size_t b = 0; b < (std::size_t)rows * nblocks; ++b) {
        state = state * 1103515245u + 12345u;
        hw[b * 34 + 0] = (unsigned char)(0x3c + (state >> 16) % 3);
        hw[b * 34 + 1] = (unsigned char)((state >> 8) & 0xff);
        for (int q = 0; q < 32; ++q) {
          state = state * 1103515245u + 12345u;
          hw[b * 34 + 2 + q] = (unsigned char)(state >> 13);
        }
      }
      CHECK(hipMemcpy(w, hw, wbytes, hipMemcpyHostToDevice));
      delete[] hw;
    }
    const int reps = 400;
    const double gbs = (double)wbytes / 1e9;
    float t0 = Time(V0, (int)((rows + 3) / 4), 128, w, x, o0, rows, cols, reps);
    float t1 = Time(V1<4>, (int)((rows + 7) / 8), 128, w, x, o1, rows, cols, reps);
    float t2 = Time(V1<8>, (int)((rows + 7) / 8), 256, w, x, o1, rows, cols, reps);
    float t3 = Time(V3, (int)((rows + 3) / 4), 128, w, x, o1, rows, cols, reps);
    std::printf("%s: %d bytes/row\n", sh.name, 34 * (int)nblocks);
    std::printf("  V0 prod : %8.2f us  %6.1f GB/s\n", t0, gbs / t0 * 1e6);
    std::printf("  V1 pair : %8.2f us  %6.1f GB/s  %+.1f%%\n", t1, gbs / t1 * 1e6, (t1 / t0 - 1.0) * 100.0);
    std::printf("  V2 p256 : %8.2f us  %6.1f GB/s  %+.1f%%\n", t2, gbs / t2 * 1e6, (t2 / t0 - 1.0) * 100.0);
    std::printf("  V3 noscl: %8.2f us  %6.1f GB/s  %+.1f%%\n", t3, gbs / t3 * 1e6, (t3 / t0 - 1.0) * 100.0);
    // Bit-exactness of V1/V2 against V0.
    CHECK(hipMemset(o0, 0, rows * sizeof(float)));
    V0<<<(int)((rows + 3) / 4), 128>>>(w, x, o0, rows, cols);
    CHECK(hipMemset(o1, 0xCD, rows * sizeof(float)));
    V1<4><<<(int)((rows + 7) / 8), 128>>>(w, x, o1, rows, cols);
    CHECK(hipDeviceSynchronize());
    float *a = new float[rows], *b = new float[rows];
    CHECK(hipMemcpy(a, o0, rows * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(b, o1, rows * 4, hipMemcpyDeviceToHost));
    std::size_t diff1 = 0;
    for (std::uint32_t r = 0; r < rows; ++r) diff1 += std::memcmp(&a[r], &b[r], 4) != 0;
    CHECK(hipMemset(o1, 0xCD, rows * sizeof(float)));
    V1<8><<<(int)((rows + 7) / 8), 256>>>(w, x, o1, rows, cols);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(b, o1, rows * 4, hipMemcpyDeviceToHost));
    std::size_t diff2 = 0;
    for (std::uint32_t r = 0; r < rows; ++r) diff2 += std::memcmp(&a[r], &b[r], 4) != 0;
    std::printf("  bitexact V1=%d V2=%d\n", (int)diff1, (int)diff2);
    delete[] a; delete[] b;
    CHECK(hipFree(w)); CHECK(hipFree(o0)); CHECK(hipFree(o1));
  }
  CHECK(hipFree(x));
  return 0;
}
