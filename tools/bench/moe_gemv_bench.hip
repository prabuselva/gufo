// Decode GEMV tier bench for qwen36_a3b (gfx1151): exact in-model shapes for
// lin_gemm_in (ssm qkv/gate/alpha/beta), routed MoE gate+up pair and down,
// shared expert, and attention qkv. Compares the committed scalar kernels
// against launch-fused (bit-exact) and vec4-load (reordered) variants.
//
//   tools/bench/build.sh moe_gemv_bench && /tmp/moe_gemv_bench

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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

// ---- committed kernels (copied verbatim) --------------------------------

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

__global__ void GemvGroupedQ8_0Pair(const Q8_0Block* __restrict__ wa,
                                    const Q8_0Block* __restrict__ wb,
                                    const std::int32_t* __restrict__ ids,
                                    std::size_t expert_stride_blocks,
                                    const float* __restrict__ x,
                                    std::uint32_t x_stride,
                                    float* __restrict__ out_a,
                                    float* __restrict__ out_b,
                                    std::uint32_t used, std::uint32_t rows,
                                    std::uint32_t cols) {
  const std::uint32_t total = used * rows;
  const std::uint32_t p = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (p >= 2U * total) {
    return;
  }
  const bool first = p < total;
  const std::uint32_t pair = first ? p : p - total;
  const std::uint32_t s = pair / rows;
  const std::uint32_t r = pair - s * rows;
  const std::uint32_t nblocks = cols / 32;
  const Q8_0Block* __restrict__ row =
      (first ? wa : wb) +
      static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
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
    if (first) {
      out_a[pair] = acc;
    } else {
      out_b[pair] = acc;
    }
  }
}

// ---- variant: up to four same-x projections in one launch (bit-exact) ---

__device__ __forceinline__ float GemvRow(const Q8_0Block* __restrict__ row,
                                         const float* __restrict__ x,
                                         std::uint32_t nblocks,
                                         std::uint32_t lane) {
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * static_cast<float>(row[b].qs[lane]) * x[b * 32 + lane];
  }
  return WarpReduceSum(acc);
}

__global__ void GemvQ8_0Multi4(const Q8_0Block* __restrict__ w0,
                               const Q8_0Block* __restrict__ w1,
                               const Q8_0Block* __restrict__ w2,
                               const Q8_0Block* __restrict__ w3,
                               float* __restrict__ o0, float* __restrict__ o1,
                               float* __restrict__ o2, float* __restrict__ o3,
                               std::uint32_t n0, std::uint32_t n01,
                               std::uint32_t n012, std::uint32_t total,
                               std::uint32_t nb0, std::uint32_t nb1,
                               std::uint32_t nb2, std::uint32_t nb3,
                               const float* __restrict__ x) {
  const std::uint32_t r = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (r >= total) {
    return;
  }
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc;
  if (r < n0) {
    acc = GemvRow(w0 + static_cast<std::size_t>(r) * nb0, x, nb0, lane);
    if (lane == 0) o0[r] = acc;
  } else if (r < n01) {
    acc = GemvRow(w1 + static_cast<std::size_t>(r - n0) * nb1, x, nb1, lane);
    if (lane == 0) o1[r - n0] = acc;
  } else if (r < n012) {
    acc = GemvRow(w2 + static_cast<std::size_t>(r - n01) * nb2, x, nb2, lane);
    if (lane == 0) o2[r - n01] = acc;
  } else {
    acc = GemvRow(w3 + static_cast<std::size_t>(r - n012) * nb3, x, nb3, lane);
    if (lane == 0) o3[r - n012] = acc;
  }
}

// ---- variant: pair kernel with x staged in LDS (shared by all 4 waves) ---

__global__ void GemvGroupedQ8_0PairLds(const Q8_0Block* __restrict__ wa,
                                       const Q8_0Block* __restrict__ wb,
                                       const std::int32_t* __restrict__ ids,
                                       std::size_t expert_stride_blocks,
                                       const float* __restrict__ x,
                                       float* __restrict__ out_a,
                                       float* __restrict__ out_b,
                                       std::uint32_t used, std::uint32_t rows,
                                       std::uint32_t cols) {
  __shared__ float sx[2048];
  for (std::uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    sx[i] = x[i];
  }
  __syncthreads();
  const std::uint32_t total = used * rows;
  const std::uint32_t p = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (p >= 2U * total) {
    return;
  }
  const bool first = p < total;
  const std::uint32_t pair = first ? p : p - total;
  const std::uint32_t s = pair / rows;
  const std::uint32_t r = pair - s * rows;
  const std::uint32_t nblocks = cols / 32;
  const Q8_0Block* __restrict__ row =
      (first ? wa : wb) +
      static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * nblocks;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t b = 0; b < nblocks; ++b) {
    const float d = __half2float(row[b].d);
    acc += d * static_cast<float>(row[b].qs[lane]) * sx[b * 32 + lane];
  }
  acc = WarpReduceSum(acc);
  if (lane == 0) {
    if (first) {
      out_a[pair] = acc;
    } else {
      out_b[pair] = acc;
    }
  }
}

// ---- variant: vec4 int8 loads (reordered accumulation) ------------------

// Lane l owns bytes [4l, 4l+3] of each 128-byte quant-block group (4 blocks).
// The uint32 load is 4 contiguous int8; scales are read as 4 halves from a
// second (deinterleaved) scale array laid out [group][lane]? No — scales are
// per block; lane l needs d of the block containing k=4l+j, i.e. block
// (4l+j)/32 = group*4 + (4l+j)/32 ... all lanes share 4 block scales.
__global__ void GemvGroupedQ8_0PairVec4(const Q8_0Block* __restrict__ wa,
                                        const Q8_0Block* __restrict__ wb,
                                        const std::int32_t* __restrict__ ids,
                                        std::size_t expert_stride_blocks,
                                        const float* __restrict__ x,
                                        std::uint32_t x_stride,
                                        float* __restrict__ out_a,
                                        float* __restrict__ out_b,
                                        std::uint32_t used, std::uint32_t rows,
                                        std::uint32_t cols) {
  const std::uint32_t total = used * rows;
  const std::uint32_t p = blockIdx.x * 4U + (threadIdx.x >> 5);
  if (p >= 2U * total) {
    return;
  }
  const bool first = p < total;
  const std::uint32_t pair = first ? p : p - total;
  const std::uint32_t s = pair / rows;
  const std::uint32_t r = pair - s * rows;
  const std::uint32_t groups = cols / 128;
  const Q8_0Block* __restrict__ row =
      (first ? wa : wb) +
      static_cast<std::size_t>(ids[s]) * expert_stride_blocks +
      static_cast<std::size_t>(r) * (cols / 32);
  const float* __restrict__ xs = x + static_cast<std::size_t>(s) * x_stride;
  const std::uint32_t lane = threadIdx.x & 31U;
  float acc = 0.0f;
#pragma unroll 4
  for (std::uint32_t g = 0; g < groups; ++g) {
    // 4 quant blocks per group; lane l reads the uint32 at byte 4l of the
    // 128-byte qs span (blocks are {d, qs[32]} so qs spans are strided 34B;
    // read per-block uint32 instead: block j covers lanes 8j..8j+7? No:
    // byte 4l lies in block (4l)/32 = l/8. Lane l reads block l/8, word
    // (4l) % 32.
    const std::uint32_t blk = g * 4U + lane / 8U;
    const std::uint32_t word = (4U * lane) % 32U;
    std::int32_t packed;
    __builtin_memcpy(&packed, &row[blk].qs[word], 4);
    const float d = __half2float(row[blk].d);
    const std::uint32_t k = blk * 32U + word;
    acc += d * static_cast<float>(static_cast<std::int8_t>(packed & 0xFF)) *
           x[k];
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
    if (first) {
      out_a[pair] = acc;
    } else {
      out_b[pair] = acc;
    }
  }
}

// ---- harness -------------------------------------------------------------

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

// Fill a Q8_0 matrix with random-but-valid data.
std::vector<Q8_0Block> MakeQ8(std::size_t elems, std::uint32_t seed) {
  std::vector<Q8_0Block> blocks(elems / 32);
  for (auto& blk : blocks) {
    float amax = 1e-6f;
    std::int8_t qs[32];
    for (int i = 0; i < 32; ++i) {
      const float v = (static_cast<int>(NextRandom(&seed) & 0xFFFF) - 32768) /
                      32768.0f;
      qs[i] = static_cast<std::int8_t>(v * 100.0f);
      amax = amax > std::abs(static_cast<float>(qs[i])) ? amax
                                                        : std::abs(static_cast<float>(qs[i]));
    }
    blk.d = __float2half(amax / 127.0f);
    for (int i = 0; i < 32; ++i) {
      blk.qs[i] = qs[i];
    }
  }
  return blocks;
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

// Time fn() over iters launches; returns ms per iteration.
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

// Time fn(i) rotating over ncopies weight sets so consecutive iterations
// never hit the same L2 lines (models the in-model cold-L2 stream).
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
  std::printf("  %-34s %7.4f ms  %6.1f GB/s\n", name, ms,
              weight_bytes / ms / 1e6);
}

constexpr double kQ8 = 34.0 / 32.0;  // bytes per element

}  // namespace

int main() {
  constexpr std::uint32_t kHidden = 2048;
  constexpr std::uint32_t kExperts = 256;
  constexpr std::uint32_t kUsed = 8;
  constexpr std::uint32_t kExpertFf = 512;

  // x activation row.
  std::vector<float> hx(kHidden);
  std::uint32_t seed = 12345;
  for (auto& v : hx) {
    v = (static_cast<int>(NextRandom(&seed) & 0xFFFF) - 32768) / 32768.0f;
  }
  Dev x(kHidden * sizeof(float));
  x.Upload(hx.data(), hx.size() * sizeof(float));

  // Expert ids: 8 distinct.
  std::int32_t ids[kUsed];
  for (std::uint32_t i = 0; i < kUsed; ++i) ids[i] = static_cast<std::int32_t>(i * 31 + 3);
  Dev dids(sizeof(ids));
  dids.Upload(ids, sizeof(ids));

  std::printf("=== lin_gemm_in: ssm qkv 8192 + gate 4096 + alpha 32 + beta 32, cols 2048 ===\n");
  {
    const std::uint32_t r0 = 8192, r1 = 4096, r2 = 32, r3 = 32;
    const double bytes = (r0 + r1 + r2 + r3) * kHidden * kQ8;
    Dev w0(r0 * kHidden * kQ8), w1(r1 * kHidden * kQ8), w2(r2 * kHidden * kQ8),
        w3(r3 * kHidden * kQ8);
    auto b0 = MakeQ8(r0 * kHidden, 1);
    auto b1 = MakeQ8(r1 * kHidden, 2);
    auto b2 = MakeQ8(r2 * kHidden, 3);
    auto b3 = MakeQ8(r3 * kHidden, 4);
    w0.Upload(b0.data(), b0.size() * sizeof(Q8_0Block));
    w1.Upload(b1.data(), b1.size() * sizeof(Q8_0Block));
    w2.Upload(b2.data(), b2.size() * sizeof(Q8_0Block));
    w3.Upload(b3.data(), b3.size() * sizeof(Q8_0Block));
    Dev o0(r0 * 4), o1(r1 * 4), o2(r2 * 4), o3(r3 * 4);
    const std::uint32_t total = r0 + r1 + r2 + r3;
    Report("4 separate Gemv launches",
           Time([&] {
             GemvQ8_0<<<(r0 + 3) / 4, 128>>>(w0.get<Q8_0Block>(), x.get<float>(),
                                             o0.get<float>(), r0, kHidden);
             GemvQ8_0<<<(r1 + 3) / 4, 128>>>(w1.get<Q8_0Block>(), x.get<float>(),
                                             o1.get<float>(), r1, kHidden);
             GemvQ8_0<<<(r2 + 3) / 4, 128>>>(w2.get<Q8_0Block>(), x.get<float>(),
                                             o2.get<float>(), r2, kHidden);
             GemvQ8_0<<<(r3 + 3) / 4, 128>>>(w3.get<Q8_0Block>(), x.get<float>(),
                                             o3.get<float>(), r3, kHidden);
           }),
           bytes);
    Report("GemvQ8_0Multi4 (1 launch)",
           Time([&] {
             GemvQ8_0Multi4<<<(total + 3) / 4, 128>>>(
                 w0.get<Q8_0Block>(), w1.get<Q8_0Block>(), w2.get<Q8_0Block>(),
                 w3.get<Q8_0Block>(), o0.get<float>(), o1.get<float>(),
                 o2.get<float>(), o3.get<float>(), r0, r0 + r1, r0 + r1 + r2,
                 total, kHidden / 32, kHidden / 32, kHidden / 32, kHidden / 32,
                 x.get<float>());
           }),
           bytes);
  }

  std::printf("=== routed gate+up pair: used=8 rows=512 cols=2048, 256 experts ===\n");
  {
    const std::uint32_t rows = kExpertFf, cols = kHidden;
    const double bytes = 2.0 * kUsed * rows * cols * kQ8;
    Dev wa(static_cast<std::size_t>(kExperts) * rows * cols * kQ8),
        wb(static_cast<std::size_t>(kExperts) * rows * cols * kQ8);
    auto ba = MakeQ8(static_cast<std::size_t>(kExperts) * rows * cols, 5);
    auto bb = MakeQ8(static_cast<std::size_t>(kExperts) * rows * cols, 6);
    wa.Upload(ba.data(), ba.size() * sizeof(Q8_0Block));
    wb.Upload(bb.data(), bb.size() * sizeof(Q8_0Block));
    Dev oa(kUsed * rows * 4), ob(kUsed * rows * 4);
    const std::size_t stride = static_cast<std::size_t>(rows) * cols / 32;
    const std::uint32_t grid = (2 * kUsed * rows + 3) / 4;
    Report("GemvGroupedQ8_0Pair (scalar)",
           Time([&] {
             GemvGroupedQ8_0Pair<<<grid, 128>>>(
                 wa.get<Q8_0Block>(), wb.get<Q8_0Block>(), dids.get<std::int32_t>(),
                 stride, x.get<float>(), 0U, oa.get<float>(), ob.get<float>(),
                 kUsed, rows, cols);
           }),
           bytes);
    Report("...Vec4 (reordered)",
           Time([&] {
             GemvGroupedQ8_0PairVec4<<<grid, 128>>>(
                 wa.get<Q8_0Block>(), wb.get<Q8_0Block>(), dids.get<std::int32_t>(),
                 stride, x.get<float>(), 0U, oa.get<float>(), ob.get<float>(),
                 kUsed, rows, cols);
           }),
           bytes);
  }

  std::printf("=== routed down: used=8 rows=2048 cols=512, x_stride=512 ===\n");
  {
    const std::uint32_t rows = kHidden, cols = kExpertFf;
    const double bytes = static_cast<double>(kUsed) * rows * cols * kQ8;
    Dev w(static_cast<std::size_t>(kExperts) * rows * cols * kQ8);
    auto bw = MakeQ8(static_cast<std::size_t>(kExperts) * rows * cols, 7);
    w.Upload(bw.data(), bw.size() * sizeof(Q8_0Block));
    Dev o(kUsed * rows * 4);
    const std::size_t stride = static_cast<std::size_t>(rows) * cols / 32;
    Report("GemvGroupedQ8_0",
           Time([&] {
             GemvGroupedQ8_0<<<(kUsed * rows + 3) / 4, 128>>>(
                 w.get<Q8_0Block>(), dids.get<std::int32_t>(), stride,
                 x.get<float>(), cols, o.get<float>(), kUsed, rows, cols);
           }),
           bytes);
  }

  std::printf("=== shared expert: gate 512 + up 512 + gate_inp 1 (cols 2048), down 2048x512 ===\n");
  {
    const std::uint32_t rf = kExpertFf;
    const double bytes = (2.0 * rf + 1 + kHidden) * 0 +
                         (2.0 * rf * kHidden + 1 * kHidden + kHidden * rf) * kQ8;
    Dev gate(rf * kHidden * kQ8), up(rf * kHidden * kQ8),
        gate_inp(kHidden * kQ8), down(kHidden * rf * kQ8);
    auto bg = MakeQ8(rf * kHidden, 8);
    auto bu = MakeQ8(rf * kHidden, 9);
    auto bi = MakeQ8(kHidden, 10);
    auto bd = MakeQ8(kHidden * rf, 11);
    gate.Upload(bg.data(), bg.size() * sizeof(Q8_0Block));
    up.Upload(bu.data(), bu.size() * sizeof(Q8_0Block));
    gate_inp.Upload(bi.data(), bi.size() * sizeof(Q8_0Block));
    down.Upload(bd.data(), bd.size() * sizeof(Q8_0Block));
    Dev og(rf * 4), ou(rf * 4), oi(4), od(kHidden * 4);
    Report("pair + down + gate_inp (3 launches)",
           Time([&] {
             GemvQ8_0<<<(rf + 3) / 4, 128>>>(gate.get<Q8_0Block>(),
                                             x.get<float>(), og.get<float>(), rf,
                                             kHidden);
             GemvQ8_0<<<(rf + 3) / 4, 128>>>(up.get<Q8_0Block>(), x.get<float>(),
                                             ou.get<float>(), rf, kHidden);
             GemvQ8_0<<<(kHidden + 3) / 4, 128>>>(down.get<Q8_0Block>(),
                                                  og.get<float>(),
                                                  od.get<float>(), kHidden, rf);
             GemvQ8_0<<<1, 128>>>(gate_inp.get<Q8_0Block>(), x.get<float>(),
                                  oi.get<float>(), 1, kHidden);
           }),
           bytes);
    Report("Multi4 gate+up+gate_inp+down*",
           Time([&] {
             const std::uint32_t total = rf + rf + 1 + kHidden;
             GemvQ8_0Multi4<<<(total + 3) / 4, 128>>>(
                 gate.get<Q8_0Block>(), up.get<Q8_0Block>(),
                 gate_inp.get<Q8_0Block>(), down.get<Q8_0Block>(),
                 og.get<float>(), ou.get<float>(), oi.get<float>(),
                 od.get<float>(), rf, rf + rf, rf + rf + 1, total,
                 kHidden / 32, kHidden / 32, kHidden / 32, rf / 32,
                 x.get<float>());
           }),
           bytes);
  }

  std::printf("=== attn qkv: q 4096 + k 512 + v 512, cols 2048 (10 layers) ===\n");
  {
    const std::uint32_t rq = 4096, rkv = 512;
    const double bytes = (rq + 2 * rkv) * kHidden * kQ8;
    Dev wq(rq * kHidden * kQ8), wk(rkv * kHidden * kQ8), wv(rkv * kHidden * kQ8);
    auto bq = MakeQ8(rq * kHidden, 12);
    auto bk = MakeQ8(rkv * kHidden, 13);
    auto bv = MakeQ8(rkv * kHidden, 14);
    wq.Upload(bq.data(), bq.size() * sizeof(Q8_0Block));
    wk.Upload(bk.data(), bk.size() * sizeof(Q8_0Block));
    wv.Upload(bv.data(), bv.size() * sizeof(Q8_0Block));
    Dev oq(rq * 4), ok(rkv * 4), ov(rkv * 4);
    Report("3 separate Gemv launches",
           Time([&] {
             GemvQ8_0<<<(rq + 3) / 4, 128>>>(wq.get<Q8_0Block>(), x.get<float>(),
                                             oq.get<float>(), rq, kHidden);
             GemvQ8_0<<<(rkv + 3) / 4, 128>>>(wk.get<Q8_0Block>(), x.get<float>(),
                                              ok.get<float>(), rkv, kHidden);
             GemvQ8_0<<<(rkv + 3) / 4, 128>>>(wv.get<Q8_0Block>(), x.get<float>(),
                                              ov.get<float>(), rkv, kHidden);
           }),
           bytes);
    Report("Multi4 (1 launch, down=wq)",
           Time([&] {
             const std::uint32_t total = rq + rkv + rkv;
             GemvQ8_0Multi4<<<(total + 3) / 4, 128>>>(
                 wq.get<Q8_0Block>(), wk.get<Q8_0Block>(), wv.get<Q8_0Block>(),
                 wq.get<Q8_0Block>(), oq.get<float>(), ok.get<float>(),
                 ov.get<float>(), nullptr, rq, rq + rkv, rq + rkv + rkv, total,
                 kHidden / 32, kHidden / 32, kHidden / 32, 1, x.get<float>());
           }),
           bytes);
  }

  std::printf("=== COLD-L2: routed gate+up pair (rotating expert ids) ===\n");
  {
    const std::uint32_t rows = kExpertFf, cols = kHidden;
    const double bytes = 2.0 * kUsed * rows * cols * kQ8;
    constexpr int kSlots = 8;
    Dev wa(static_cast<std::size_t>(kExperts) * rows * cols * kQ8),
        wb(static_cast<std::size_t>(kExperts) * rows * cols * kQ8);
    auto ba = MakeQ8(static_cast<std::size_t>(kExperts) * rows * cols, 5);
    auto bb = MakeQ8(static_cast<std::size_t>(kExperts) * rows * cols, 6);
    wa.Upload(ba.data(), ba.size() * sizeof(Q8_0Block));
    wb.Upload(bb.data(), bb.size() * sizeof(Q8_0Block));
    Dev oa(kUsed * rows * 4), ob(kUsed * rows * 4);
    std::int32_t slot_ids[kSlots * kUsed];
    for (int c = 0; c < kSlots; ++c) {
      for (std::uint32_t i = 0; i < kUsed; ++i) {
        slot_ids[c * kUsed + i] =
            static_cast<std::int32_t>((c * kUsed + i * 31 + 3) % kExperts);
      }
    }
    Dev dslots(sizeof(slot_ids));
    dslots.Upload(slot_ids, sizeof(slot_ids));
    const std::size_t stride = static_cast<std::size_t>(rows) * cols / 32;
    const std::uint32_t grid = (2 * kUsed * rows + 3) / 4;
    Report("scalar cold",
           TimeCold(
               [&](int c) {
                 GemvGroupedQ8_0Pair<<<grid, 128>>>(
                     wa.get<Q8_0Block>(), wb.get<Q8_0Block>(),
                     dslots.get<std::int32_t>() + c * kUsed, stride,
                     x.get<float>(), 0U, oa.get<float>(), ob.get<float>(), kUsed,
                     rows, cols);
               },
               kSlots),
           bytes);
    Report("LDS-x cold",
           TimeCold(
               [&](int c) {
                 GemvGroupedQ8_0PairLds<<<grid, 128>>>(
                     wa.get<Q8_0Block>(), wb.get<Q8_0Block>(),
                     dslots.get<std::int32_t>() + c * kUsed, stride,
                     x.get<float>(), oa.get<float>(), ob.get<float>(), kUsed,
                     rows, cols);
               },
               kSlots),
           bytes);
    Report("vec4 cold",
           TimeCold(
               [&](int c) {
                 GemvGroupedQ8_0PairVec4<<<grid, 128>>>(
                     wa.get<Q8_0Block>(), wb.get<Q8_0Block>(),
                     dslots.get<std::int32_t>() + c * kUsed, stride,
                     x.get<float>(), 0U, oa.get<float>(), ob.get<float>(), kUsed,
                     rows, cols);
               },
               kSlots),
           bytes);
  }

  std::printf("=== COLD-L2: routed down (rotating expert ids) ===\n");
  {
    const std::uint32_t rows = kHidden, cols = kExpertFf;
    const double bytes = static_cast<double>(kUsed) * rows * cols * kQ8;
    constexpr int kSlots = 8;
    Dev w(static_cast<std::size_t>(kExperts) * rows * cols * kQ8);
    auto bw = MakeQ8(static_cast<std::size_t>(kExperts) * rows * cols, 7);
    w.Upload(bw.data(), bw.size() * sizeof(Q8_0Block));
    Dev o(kUsed * rows * 4);
    std::int32_t slot_ids[kSlots * kUsed];
    for (int c = 0; c < kSlots; ++c) {
      for (std::uint32_t i = 0; i < kUsed; ++i) {
        slot_ids[c * kUsed + i] =
            static_cast<std::int32_t>((c * kUsed + i * 31 + 3) % kExperts);
      }
    }
    Dev dslots(sizeof(slot_ids));
    dslots.Upload(slot_ids, sizeof(slot_ids));
    const std::size_t stride = static_cast<std::size_t>(rows) * cols / 32;
    Report("scalar cold",
           TimeCold(
               [&](int c) {
                 GemvGroupedQ8_0<<<(kUsed * rows + 3) / 4, 128>>>(
                     w.get<Q8_0Block>(), dslots.get<std::int32_t>() + c * kUsed,
                     stride, x.get<float>(), cols, o.get<float>(), kUsed, rows,
                     cols);
               },
               kSlots),
           bytes);
  }

  std::printf("=== COLD-L2: lin_gemm_in (6 row-window copies) ===\n");
  {
    const std::uint32_t r0 = 8192, r1 = 4096, r2 = 32, r3 = 32;
    const double bytes = (r0 + r1 + r2 + r3) * kHidden * kQ8;
    constexpr int kSlots = 6;
    Dev w0(static_cast<std::size_t>(r0) * kSlots * kHidden * kQ8),
        w1(static_cast<std::size_t>(r1) * kSlots * kHidden * kQ8),
        w2(static_cast<std::size_t>(r2) * kSlots * kHidden * kQ8),
        w3(static_cast<std::size_t>(r3) * kSlots * kHidden * kQ8);
    auto b0 = MakeQ8(static_cast<std::size_t>(r0) * kSlots * kHidden, 1);
    auto b1 = MakeQ8(static_cast<std::size_t>(r1) * kSlots * kHidden, 2);
    auto b2 = MakeQ8(static_cast<std::size_t>(r2) * kSlots * kHidden, 3);
    auto b3 = MakeQ8(static_cast<std::size_t>(r3) * kSlots * kHidden, 4);
    w0.Upload(b0.data(), b0.size() * sizeof(Q8_0Block));
    w1.Upload(b1.data(), b1.size() * sizeof(Q8_0Block));
    w2.Upload(b2.data(), b2.size() * sizeof(Q8_0Block));
    w3.Upload(b3.data(), b3.size() * sizeof(Q8_0Block));
    Dev o0(r0 * 4), o1(r1 * 4), o2(r2 * 4), o3(r3 * 4);
    const std::size_t nb0 = r0 * kHidden / 32, nb1 = r1 * kHidden / 32,
                      nb2 = r2 * kHidden / 32, nb3 = r3 * kHidden / 32;
    Report("4 launches cold",
           TimeCold(
               [&](int c) {
                 GemvQ8_0<<<(r0 + 3) / 4, 128>>>(
                     w0.get<Q8_0Block>() + c * nb0, x.get<float>(),
                     o0.get<float>(), r0, kHidden);
                 GemvQ8_0<<<(r1 + 3) / 4, 128>>>(
                     w1.get<Q8_0Block>() + c * nb1, x.get<float>(),
                     o1.get<float>(), r1, kHidden);
                 GemvQ8_0<<<(r2 + 3) / 4, 128>>>(
                     w2.get<Q8_0Block>() + c * nb2, x.get<float>(),
                     o2.get<float>(), r2, kHidden);
                 GemvQ8_0<<<(r3 + 3) / 4, 128>>>(
                     w3.get<Q8_0Block>() + c * nb3, x.get<float>(),
                     o3.get<float>(), r3, kHidden);
               },
               kSlots),
           bytes);
    const std::uint32_t total = r0 + r1 + r2 + r3;
    Report("Multi4 cold",
           TimeCold(
               [&](int c) {
                 GemvQ8_0Multi4<<<(total + 3) / 4, 128>>>(
                     w0.get<Q8_0Block>() + c * nb0,
                     w1.get<Q8_0Block>() + c * nb1,
                     w2.get<Q8_0Block>() + c * nb2,
                     w3.get<Q8_0Block>() + c * nb3, o0.get<float>(),
                     o1.get<float>(), o2.get<float>(), o3.get<float>(), r0,
                     r0 + r1, r0 + r1 + r2, total, kHidden / 32, kHidden / 32,
                     kHidden / 32, kHidden / 32, x.get<float>());
               },
               kSlots),
           bytes);
  }

  return 0;
}