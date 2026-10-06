// Decode GEMV tier bench for gemma4 (gfx1151): exact in-model shapes for the
// dense projections + lm_head (GemvQ8_0), the routed MoE grouped decode GEMV
// (GemvGroupedQ8_0, gate/up interleaved as one 2*expert_ff-row tensor and the
// down projection) and the fused attention qkv (GemvQ8_0Multi4). Compares the
// committed scalar kernels against vec4-load variants (4 int8 per lane per
// group of 4 quant blocks; reordered accumulation).
//
// Every vec4 kernel is checked against a CPU double reference over the exactly
// dequantized weights, so the reorder is validated, not assumed.
//
//   tools/bench/build.sh gemma4_gemv_bench && /tmp/gemma4_gemv_bench

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down(v, offset);
  }
  return v;
}

// ---- committed scalar kernels (copied verbatim) -------------------------

__global__ void GemvQ8_0(const Q8_0Block* __restrict__ w,
                         const float* __restrict__ x, float* __restrict__ out,
                         std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t nblocks = cols / 32;
  const Q8_0Block* __restrict__ row = w + static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * static_cast<float>(row[b].qs[lane]) * x[b * 32 + lane];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

__global__ void GemvGroupedQ8_0(const Q8_0Block* __restrict__ w,
                                const std::int32_t* __restrict__ ids,
                                std::size_t expert_stride_blocks,
                                const float* __restrict__ x,
                                std::uint32_t x_stride, float* __restrict__ out,
                                std::uint32_t used, std::uint32_t rows,
                                std::uint32_t cols) {
  const std::uint32_t pair = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (pair >= used * rows) {
    return;
  }
  const std::uint32_t s = pair / rows;
  const std::uint32_t r = pair - s * rows;
  const std::uint32_t nblocks = cols / 32;
  const Q8_0Block* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * nblocks;
  const float* __restrict__ xs = x + static_cast<std::size_t>(s) * x_stride;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * static_cast<float>(row[b].qs[lane]) * xs[b * 32 + lane];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[pair] = acc;
  }
}

struct Multi4Args {
  const Q8_0Block* w[4];
  float* out[4];
  std::uint32_t off[5];
  std::uint32_t nb[4];
};

__global__ void GemvQ8_0Multi4(Multi4Args a, const float* __restrict__ x) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= a.off[4]) {
    return;
  }
  std::uint32_t t = 0;
  while (r >= a.off[t + 1]) {
    ++t;
  }
  const std::uint32_t row_index = r - a.off[t];
  const Q8_0Block* __restrict__ row =
      a.w[t] + static_cast<std::size_t>(row_index) * a.nb[t];
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < a.nb[t]; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * static_cast<float>(row[b].qs[lane]) * x[b * 32 + lane];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    a.out[t][row_index] = acc;
  }
}

// ---- vec4 variants (reordered accumulation) -----------------------------
//
// Lane l reads the uint32 at byte 4l of the 128-byte qs span of a group of 4
// quant blocks: block = g*4 + l/8, word = (4l) % 32. Each block's 32 codes are
// read by 8 lanes x 4 bytes; the warp issues one 128-byte transaction per
// group instead of four 32-byte ones. The per-block scale d is applied to the
// four codes of the block the lane read, so the dequantization is exact; only
// the summation order differs from the scalar kernel.

__global__ void GemvQ8_0Vec4(const Q8_0Block* __restrict__ w,
                             const float* __restrict__ x,
                             float* __restrict__ out, std::uint32_t rows,
                             std::uint32_t cols) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= rows) {
    return;
  }
  const std::uint32_t groups = cols / 128;
  const Q8_0Block* __restrict__ row = w + static_cast<std::size_t>(r) * (cols / 32);
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
    acc += d * static_cast<float>(static_cast<std::int8_t>((packed >> 8) & 0xFF)) *
           x[k + 1];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 16) & 0xFF)) *
           x[k + 2];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 24) & 0xFF)) *
           x[k + 3];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[r] = acc;
  }
}

__global__ void GemvGroupedQ8_0Vec4(const Q8_0Block* __restrict__ w,
                                    const std::int32_t* __restrict__ ids,
                                    std::size_t expert_stride_blocks,
                                    const float* __restrict__ x,
                                    std::uint32_t x_stride,
                                    float* __restrict__ out, std::uint32_t used,
                                    std::uint32_t rows, std::uint32_t cols) {
  const std::uint32_t pair = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (pair >= used * rows) {
    return;
  }
  const std::uint32_t s = pair / rows;
  const std::uint32_t r = pair - s * rows;
  const std::uint32_t groups = cols / 128;
  const Q8_0Block* __restrict__ row =
      w + static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * (cols / 32);
  const float* __restrict__ xs = x + static_cast<std::size_t>(s) * x_stride;
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
    acc += d * static_cast<float>(static_cast<std::int8_t>(packed & 0xFF)) *
           xs[k];
    acc += d * static_cast<float>(static_cast<std::int8_t>((packed >> 8) & 0xFF)) *
           xs[k + 1];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 16) & 0xFF)) *
           xs[k + 2];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 24) & 0xFF)) *
           xs[k + 3];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[pair] = acc;
  }
}

__global__ void GemvQ8_0Multi4Vec4(Multi4Args a, const float* __restrict__ x) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= a.off[4]) {
    return;
  }
  std::uint32_t t = 0;
  while (r >= a.off[t + 1]) {
    ++t;
  }
  const std::uint32_t row_index = r - a.off[t];
  const std::uint32_t groups = a.nb[t] / 4U;
  const Q8_0Block* __restrict__ row =
      a.w[t] + static_cast<std::size_t>(row_index) * a.nb[t];
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
    acc += d * static_cast<float>(static_cast<std::int8_t>((packed >> 8) & 0xFF)) *
           x[k + 1];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 16) & 0xFF)) *
           x[k + 2];
    acc += d *
           static_cast<float>(static_cast<std::int8_t>((packed >> 24) & 0xFF)) *
           x[k + 3];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    a.out[t][row_index] = acc;
  }
}

// ---- harness ------------------------------------------------------------

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

std::uint32_t NextRandom(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

// Random Q8_0 blocks plus the exactly-dequantized values for the reference.
struct Q8 {
  std::vector<Q8_0Block> blocks;
  std::vector<float> values;  // elems, dequantized
};

Q8 MakeQ8(std::size_t elems, std::uint32_t seed) {
  Q8 out;
  out.blocks.resize(elems / 32);
  out.values.resize(elems);
  for (std::size_t bi = 0; bi < out.blocks.size(); ++bi) {
    float amax = 1e-6f;
    std::int8_t qs[32];
    for (int i = 0; i < 32; ++i) {
      const float v = (static_cast<int>(NextRandom(&seed) & 0xFFFF) - 32768) /
                      32768.0f;
      qs[i] = static_cast<std::int8_t>(v * 100.0f);
      amax = amax > std::abs(static_cast<float>(qs[i])) ? amax
                                                        : std::abs(static_cast<float>(qs[i]));
    }
    const __half d = __float2half(amax / 127.0f);
    const float df = __half2float(d);
    for (int i = 0; i < 32; ++i) {
      out.blocks[bi].d = d;
      out.blocks[bi].qs[i] = qs[i];
      out.values[bi * 32 + i] = df * static_cast<float>(qs[i]);
    }
  }
  return out;
}

struct Dev {
  void* p = nullptr;
  explicit Dev(std::size_t bytes) { Check(hipMalloc(&p, bytes), "malloc"); }
  ~Dev() { hipFree(p); }
  template <typename T>
  T* get() {
    return static_cast<T*>(p);
  }
  void Upload(const void* src, std::size_t bytes) {
    Check(hipMemcpy(p, src, bytes, hipMemcpyHostToDevice), "upload");
  }
};

template <typename F>
double Time(F&& fn, int iters = 200) {
  for (int i = 0; i < 20; ++i) fn();
  Check(hipDeviceSynchronize(), "sync");
  hipEvent_t a, b;
  hipEventCreate(&a);
  hipEventCreate(&b);
  hipEventRecord(a);
  for (int i = 0; i < iters; ++i) fn();
  hipEventRecord(b);
  hipEventSynchronize(b);
  float ms = 0.0f;
  hipEventElapsedTime(&ms, a, b);
  hipEventDestroy(a);
  hipEventDestroy(b);
  return ms / iters;
}

template <typename F>
double TimeCold(F&& fn, int ncopies, int iters = 100) {
  for (int i = 0; i < ncopies; ++i) fn(i);
  Check(hipDeviceSynchronize(), "sync");
  hipEvent_t a, b;
  hipEventCreate(&a);
  hipEventCreate(&b);
  hipEventRecord(a);
  for (int i = 0; i < iters; ++i) fn(i % ncopies);
  hipEventRecord(b);
  hipEventSynchronize(b);
  float ms = 0.0f;
  hipEventElapsedTime(&ms, a, b);
  hipEventDestroy(a);
  hipEventDestroy(b);
  return ms / iters;
}

void Report(const char* name, double ms, double weight_bytes) {
  std::printf("  %-30s %7.4f ms  %6.1f GB/s\n", name, ms,
              weight_bytes / ms / 1e6);
}

double WorstRelative(const std::vector<float>& ref,
                     const std::vector<float>& got) {
  double worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double r = std::abs(static_cast<double>(ref[i]));
    const double d = std::abs(static_cast<double>(ref[i]) - got[i]);
    const double rel = d / (r > 1e-6 ? r : 1e-6);
    if (rel > worst) {
      worst = rel;
    }
  }
  return worst;
}

std::vector<float> Download(void* p, std::size_t n) {
  std::vector<float> h(n);
  Check(hipMemcpy(h.data(), p, n * sizeof(float), hipMemcpyDeviceToHost),
        "download");
  return h;
}

constexpr double kQ8 = 34.0 / 32.0;

// Reference dot for a dense [rows x cols] matrix against x.
std::vector<float> RefDense(const std::vector<float>& vals, std::uint32_t rows,
                            std::uint32_t cols, const std::vector<float>& x) {
  std::vector<float> ref(rows);
  for (std::uint32_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    const float* wr = vals.data() + static_cast<std::size_t>(r) * cols;
    for (std::uint32_t i = 0; i < cols; ++i) {
      acc += static_cast<double>(wr[i]) * x[i];
    }
    ref[r] = static_cast<float>(acc);
  }
  return ref;
}

}  // namespace

int main() {
  // gemma4 geometry.
  constexpr std::uint32_t kHidden = 2816;
  // 16 stacked experts is enough: only kUsed=8 rows are read per launch
  // (2.16 GB >> L2), so the cold-L2 behavior and the expert-id indirection
  // match the in-model 128-expert stack while keeping generation/upload fast.
  constexpr std::uint32_t kExperts = 16;
  constexpr std::uint32_t kUsed = 8;
  constexpr std::uint32_t kExpertFf = 1408;

  std::vector<float> hx(kHidden);
  std::uint32_t seed = 12345;
  for (auto& v : hx) {
    v = (static_cast<int>(NextRandom(&seed) & 0xFFFF) - 32768) / 32768.0f;
  }
  Dev x(kHidden * sizeof(float));
  x.Upload(hx.data(), hx.size() * sizeof(float));

  std::int32_t ids[kUsed];
  for (std::uint32_t i = 0; i < kUsed; ++i) {
    ids[i] = static_cast<std::int32_t>((i * 31 + 3) % kExperts);
  }
  Dev dids(sizeof(ids));
  dids.Upload(ids, sizeof(ids));

  bool ok = true;

  // --- dense lm_head (reduced rows for a fast reference; throughput is
  // per-row and saturates well below the full 262144). ---
  std::printf("=== dense lm_head-ish: rows=8192 cols=2816 ===\n");
  {
    const std::uint32_t rows = 8192, cols = kHidden;
    const double bytes = rows * cols * kQ8;
    auto w = MakeQ8(static_cast<std::size_t>(rows) * cols, 101);
    Dev dw(w.blocks.size() * sizeof(Q8_0Block));
    dw.Upload(w.blocks.data(), w.blocks.size() * sizeof(Q8_0Block));
    Dev o(rows * 4);
    GemvQ8_0<<<(rows + 3) / 4, 128>>>(dw.get<Q8_0Block>(), x.get<float>(),
                                      o.get<float>(), rows, cols);
    Check(hipDeviceSynchronize(), "scalar sync");
    auto ref = RefDense(w.values, rows, cols, hx);
    double ws = WorstRelative(ref, Download(o.get<float>(), rows));
    GemvQ8_0Vec4<<<(rows + 3) / 4, 128>>>(dw.get<Q8_0Block>(), x.get<float>(),
                                          o.get<float>(), rows, cols);
    Check(hipDeviceSynchronize(), "vec4 sync");
    double wv = WorstRelative(ref, Download(o.get<float>(), rows));
    std::printf("  scalar worst-rel %.3e   vec4 worst-rel %.3e\n", ws, wv);
    ok = ok && (ws <= 2e-3) && (wv <= 2e-3);
    Report("scalar", Time([&] {
             GemvQ8_0<<<(rows + 3) / 4, 128>>>(dw.get<Q8_0Block>(),
                                               x.get<float>(), o.get<float>(),
                                               rows, cols);
           }), bytes);
    Report("vec4", Time([&] {
             GemvQ8_0Vec4<<<(rows + 3) / 4, 128>>>(dw.get<Q8_0Block>(),
                                                   x.get<float>(), o.get<float>(),
                                                   rows, cols);
           }), bytes);
  }

  // --- dense attn_output projection: rows=2816 cols=2816 ---
  std::printf("=== dense proj: rows=2816 cols=2816 ===\n");
  {
    const std::uint32_t rows = kHidden, cols = kHidden;
    const double bytes = rows * cols * kQ8;
    auto w = MakeQ8(static_cast<std::size_t>(rows) * cols, 202);
    Dev dw(w.blocks.size() * sizeof(Q8_0Block));
    dw.Upload(w.blocks.data(), w.blocks.size() * sizeof(Q8_0Block));
    Dev o(rows * 4);
    GemvQ8_0<<<(rows + 3) / 4, 128>>>(dw.get<Q8_0Block>(), x.get<float>(),
                                      o.get<float>(), rows, cols);
    Check(hipDeviceSynchronize(), "scalar sync");
    auto ref = RefDense(w.values, rows, cols, hx);
    double ws = WorstRelative(ref, Download(o.get<float>(), rows));
    GemvQ8_0Vec4<<<(rows + 3) / 4, 128>>>(dw.get<Q8_0Block>(), x.get<float>(),
                                          o.get<float>(), rows, cols);
    Check(hipDeviceSynchronize(), "vec4 sync");
    double wv = WorstRelative(ref, Download(o.get<float>(), rows));
    std::printf("  scalar worst-rel %.3e   vec4 worst-rel %.3e\n", ws, wv);
    ok = ok && (ws <= 2e-3) && (wv <= 2e-3);
    Report("scalar", Time([&] {
             GemvQ8_0<<<(rows + 3) / 4, 128>>>(dw.get<Q8_0Block>(),
                                               x.get<float>(), o.get<float>(),
                                               rows, cols);
           }), bytes);
    Report("vec4", Time([&] {
             GemvQ8_0Vec4<<<(rows + 3) / 4, 128>>>(dw.get<Q8_0Block>(),
                                                   x.get<float>(), o.get<float>(),
                                                   rows, cols);
           }), bytes);
  }

  // --- grouped gate_up: used=8 rows=2*1408 cols=2816, 128 experts ---
  std::printf("=== grouped gate_up: used=8 rows=2816 cols=2816, 16 experts ===\n");
  {
    const std::uint32_t rows = 2 * kExpertFf, cols = kHidden;
    const double bytes = static_cast<double>(kUsed) * rows * cols * kQ8;
    auto w = MakeQ8(static_cast<std::size_t>(kExperts) * rows * cols, 303);
    Dev dw(w.blocks.size() * sizeof(Q8_0Block));
    dw.Upload(w.blocks.data(), w.blocks.size() * sizeof(Q8_0Block));
    Dev o(kUsed * rows * 4);
    const std::size_t stride = static_cast<std::size_t>(rows) * cols / 32;
    const std::uint32_t grid = (kUsed * rows + 3) / 4;
    // Reference over the 8 selected experts.
    std::vector<float> ref(kUsed * rows);
    for (std::uint32_t s = 0; s < kUsed; ++s) {
      const float* we = w.values.data() +
                        static_cast<std::size_t>(ids[s]) * rows * cols;
      for (std::uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        const float* wr = we + static_cast<std::size_t>(r) * cols;
        for (std::uint32_t i = 0; i < cols; ++i) {
          acc += static_cast<double>(wr[i]) * hx[i];
        }
        ref[s * rows + r] = static_cast<float>(acc);
      }
    }
    GemvGroupedQ8_0<<<grid, 128>>>(dw.get<Q8_0Block>(), dids.get<std::int32_t>(),
                                   stride, x.get<float>(), 0U, o.get<float>(),
                                   kUsed, rows, cols);
    Check(hipDeviceSynchronize(), "scalar sync");
    double ws = WorstRelative(ref, Download(o.get<float>(), ref.size()));
    GemvGroupedQ8_0Vec4<<<grid, 128>>>(dw.get<Q8_0Block>(),
                                       dids.get<std::int32_t>(), stride,
                                       x.get<float>(), 0U, o.get<float>(), kUsed,
                                       rows, cols);
    Check(hipDeviceSynchronize(), "vec4 sync");
    double wv = WorstRelative(ref, Download(o.get<float>(), ref.size()));
    std::printf("  scalar worst-rel %.3e   vec4 worst-rel %.3e  (test tol 1e-4)\n",
                ws, wv);
    ok = ok && (ws <= 1e-4) && (wv <= 1e-4);
    Report("scalar", Time([&] {
             GemvGroupedQ8_0<<<grid, 128>>>(dw.get<Q8_0Block>(),
                                            dids.get<std::int32_t>(), stride,
                                            x.get<float>(), 0U, o.get<float>(),
                                            kUsed, rows, cols);
           }), bytes);
    Report("vec4", Time([&] {
             GemvGroupedQ8_0Vec4<<<grid, 128>>>(dw.get<Q8_0Block>(),
                                                dids.get<std::int32_t>(), stride,
                                                x.get<float>(), 0U, o.get<float>(),
                                                kUsed, rows, cols);
           }), bytes);
  }

  // --- grouped down: used=8 rows=2816 cols=1408, 128 experts ---
  std::printf("=== grouped down: used=8 rows=2816 cols=1408, 16 experts ===\n");
  {
    const std::uint32_t rows = kHidden, cols = kExpertFf;
    const double bytes = static_cast<double>(kUsed) * rows * cols * kQ8;
    auto w = MakeQ8(static_cast<std::size_t>(kExperts) * rows * cols, 404);
    Dev dw(w.blocks.size() * sizeof(Q8_0Block));
    dw.Upload(w.blocks.data(), w.blocks.size() * sizeof(Q8_0Block));
    Dev o(kUsed * rows * 4);
    const std::size_t stride = static_cast<std::size_t>(rows) * cols / 32;
    const std::uint32_t grid = (kUsed * rows + 3) / 4;
    std::vector<float> ref(kUsed * rows);
    for (std::uint32_t s = 0; s < kUsed; ++s) {
      const float* we = w.values.data() +
                        static_cast<std::size_t>(ids[s]) * rows * cols;
      for (std::uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        const float* wr = we + static_cast<std::size_t>(r) * cols;
        for (std::uint32_t i = 0; i < cols; ++i) {
          acc += static_cast<double>(wr[i]) * hx[i];
        }
        ref[s * rows + r] = static_cast<float>(acc);
      }
    }
    GemvGroupedQ8_0<<<grid, 128>>>(dw.get<Q8_0Block>(), dids.get<std::int32_t>(),
                                   stride, x.get<float>(), cols, o.get<float>(),
                                   kUsed, rows, cols);
    Check(hipDeviceSynchronize(), "scalar sync");
    double ws = WorstRelative(ref, Download(o.get<float>(), ref.size()));
    GemvGroupedQ8_0Vec4<<<grid, 128>>>(dw.get<Q8_0Block>(),
                                       dids.get<std::int32_t>(), stride,
                                       x.get<float>(), cols, o.get<float>(),
                                       kUsed, rows, cols);
    Check(hipDeviceSynchronize(), "vec4 sync");
    double wv = WorstRelative(ref, Download(o.get<float>(), ref.size()));
    std::printf("  scalar worst-rel %.3e   vec4 worst-rel %.3e  (test tol 1e-4)\n",
                ws, wv);
    ok = ok && (ws <= 1e-4) && (wv <= 1e-4);
    Report("scalar", Time([&] {
             GemvGroupedQ8_0<<<grid, 128>>>(dw.get<Q8_0Block>(),
                                            dids.get<std::int32_t>(), stride,
                                            x.get<float>(), cols, o.get<float>(),
                                            kUsed, rows, cols);
           }), bytes);
    Report("vec4", Time([&] {
             GemvGroupedQ8_0Vec4<<<grid, 128>>>(dw.get<Q8_0Block>(),
                                                dids.get<std::int32_t>(), stride,
                                                x.get<float>(), cols,
                                                o.get<float>(), kUsed, rows, cols);
           }), bytes);
  }

  // --- fused qkv Multi4: q 8192 + k 1024 + v 1024, cols 2816 ---
  std::printf("=== fused qkv Multi4: q 8192 + k 1024 + v 1024, cols 2816 ===\n");
  {
    const std::uint32_t rq = 8192, rk = 1024, rv = 1024, cols = kHidden;
    const std::uint32_t total = rq + rk + rv;
    const double bytes = static_cast<double>(total) * cols * kQ8;
    auto wq = MakeQ8(static_cast<std::size_t>(rq) * cols, 505);
    auto wk = MakeQ8(static_cast<std::size_t>(rk) * cols, 606);
    auto wv = MakeQ8(static_cast<std::size_t>(rv) * cols, 707);
    Dev dq(wq.blocks.size() * sizeof(Q8_0Block)),
        dk(wk.blocks.size() * sizeof(Q8_0Block)),
        dv(wv.blocks.size() * sizeof(Q8_0Block));
    dq.Upload(wq.blocks.data(), wq.blocks.size() * sizeof(Q8_0Block));
    dk.Upload(wk.blocks.data(), wk.blocks.size() * sizeof(Q8_0Block));
    dv.Upload(wv.blocks.data(), wv.blocks.size() * sizeof(Q8_0Block));
    Dev oq(rq * 4), ok_(rk * 4), ov(rv * 4);
    Multi4Args a{};
    a.w[0] = dq.get<Q8_0Block>();
    a.w[1] = dk.get<Q8_0Block>();
    a.w[2] = dv.get<Q8_0Block>();
    a.w[3] = dq.get<Q8_0Block>();
    a.out[0] = oq.get<float>();
    a.out[1] = ok_.get<float>();
    a.out[2] = ov.get<float>();
    a.out[3] = ov.get<float>();
    a.off[0] = 0;
    a.off[1] = rq;
    a.off[2] = rq + rk;
    a.off[3] = rq + rk + rv;
    a.off[4] = total;
    a.nb[0] = a.nb[1] = a.nb[2] = a.nb[3] = cols / 32;
    auto refq = RefDense(wq.values, rq, cols, hx);
    auto refk = RefDense(wk.values, rk, cols, hx);
    auto refv = RefDense(wv.values, rv, cols, hx);
    GemvQ8_0Multi4<<<(total + 3) / 4, 128>>>(a, x.get<float>());
    Check(hipDeviceSynchronize(), "scalar sync");
    double ws = WorstRelative(refq, Download(oq.get<float>(), rq));
    ws = std::max(ws, WorstRelative(refk, Download(ok_.get<float>(), rk)));
    ws = std::max(ws, WorstRelative(refv, Download(ov.get<float>(), rv)));
    GemvQ8_0Multi4Vec4<<<(total + 3) / 4, 128>>>(a, x.get<float>());
    Check(hipDeviceSynchronize(), "vec4 sync");
    double wv2 = WorstRelative(refq, Download(oq.get<float>(), rq));
    wv2 = std::max(wv2, WorstRelative(refk, Download(ok_.get<float>(), rk)));
    wv2 = std::max(wv2, WorstRelative(refv, Download(ov.get<float>(), rv)));
    std::printf("  scalar worst-rel %.3e   vec4 worst-rel %.3e\n", ws, wv2);
    ok = ok && (ws <= 2e-3) && (wv2 <= 2e-3);
    Report("scalar", Time([&] {
             GemvQ8_0Multi4<<<(total + 3) / 4, 128>>>(a, x.get<float>());
           }), bytes);
    Report("vec4", Time([&] {
             GemvQ8_0Multi4Vec4<<<(total + 3) / 4, 128>>>(a, x.get<float>());
           }), bytes);
  }

  std::printf("\n%s\n", ok ? "ALL CORRECTNESS PASS" : "CORRECTNESS FAIL");
  return ok ? 0 : 1;
}