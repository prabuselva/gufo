// Standalone A/B for the Ling3 MLA decode partial: scalar kernel vs the
// fused four-head group kernel, both checked against the fp64 host oracle.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include "src/models/ling3_flash/kernels/rocm/attention.hip.cpp"

namespace q = gufo::models::ling3_flash::rocm;

namespace {

constexpr std::uint32_t kMaxStageF4 = 5U;  // ceil(8 * 144 / 256)

// Chunk-identical clone of MlaDecodePartialGroupKernel with double-buffered
// staging: the next chunk's global loads are issued (into registers) and the
// store lands in the other LDS buffer while the current chunk is computed.
// All reductions, chunk sizes, thread counts and store orders are untouched.
__global__ void GroupDoubleBufKernel(
    const float* __restrict__ q_abs, const float* __restrict__ kv,
    const float* __restrict__ kpe, std::uint32_t n_pos, std::uint32_t tile,
    float scale, float* __restrict__ workspace, std::uint32_t heads,
    std::uint32_t kv_rank, std::uint32_t rotary) {
  __shared__ float s_row[2][q::kDecodeRows][q::kDecodeRowStride];
  __shared__ float s_scores[q::kDecodeGroup][q::kDecodeMaxTile];
  __shared__ float s_red[q::kDecodeGroup][32];
  __shared__ float s_meta[q::kDecodeGroup][2];
  const std::uint32_t row_dim = kv_rank + rotary;
  const std::uint32_t row4 = row_dim >> 2U;
  const std::uint32_t h0 = blockIdx.x * q::kDecodeGroup;
  const std::uint32_t tile_idx = blockIdx.y;
  const std::uint32_t tile_begin = tile_idx * tile;
  const std::uint32_t tile_len = min(tile, n_pos - tile_begin);
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t kv4 = kv_rank >> 2U;
  const std::uint32_t half4 = kv4 + (rotary >> 2U);

  float4 reg[2][kMaxStageF4];
  const auto fetch = [&](std::uint32_t base, std::uint32_t rows, int b) {
    const std::size_t first_row = static_cast<std::size_t>(tile_begin) + base;
#pragma unroll
    for (std::uint32_t i = 0; i < kMaxStageF4; ++i) {
      const std::uint32_t p = tid + i * blockDim.x;
      if (p >= rows * half4) {
        continue;
      }
      const std::uint32_t t = p / half4;
      const std::uint32_t c = p - t * half4;
      reg[b][i] = (c < kv4)
                      ? reinterpret_cast<const float4*>(kv +
                                                        (first_row + t) *
                                                            kv_rank)[c]
                      : reinterpret_cast<const float4*>(
                            kpe + (first_row + t) * rotary)[c - kv4];
    }
  };
  const auto store = [&](std::uint32_t rows, int b) {
#pragma unroll
    for (std::uint32_t i = 0; i < kMaxStageF4; ++i) {
      const std::uint32_t p = tid + i * blockDim.x;
      if (p >= rows * half4) {
        continue;
      }
      const std::uint32_t t = p / half4;
      const std::uint32_t c = p - t * half4;
      reinterpret_cast<float4*>(&s_row[b][t][0])[c] = reg[b][i];
    }
  };

  auto chunk_rows = [&](std::uint32_t base) {
    return tile_len - base < q::kDecodeRows ? tile_len - base
                                            : q::kDecodeRows;
  };

  uint2 chunks;
  chunks.x = (tile_len + q::kDecodeRows - 1U) / q::kDecodeRows;
  fetch(0U, chunk_rows(0U), 0);
  // Pass A.
  for (std::uint32_t c = 0; c < chunks.x; ++c) {
    const std::uint32_t base = c * q::kDecodeRows;
    const std::uint32_t rows = chunk_rows(base);
    if (c + 1U < chunks.x) {
      fetch(base + q::kDecodeRows, chunk_rows(base + q::kDecodeRows),
            static_cast<int>(c + 1U) & 1);
    }
    store(rows, static_cast<int>(c) & 1);
    __syncthreads();
    const float* buf = &s_row[c & 1][0][0];
    const std::uint32_t jobs = rows * q::kDecodeGroup;
    for (std::uint32_t job = tid >> 5U; job < jobs; job += blockDim.x >> 5U) {
      const std::uint32_t t = job >> 2U;
      const std::uint32_t g = job & 3U;
      const float4* q = reinterpret_cast<const float4*>(
          q_abs + static_cast<std::size_t>(h0 + g) * row_dim);
      const float4* row = reinterpret_cast<const float4*>(buf + t * q::kDecodeRowStride);
      float pacc = 0.0F;
      for (std::uint32_t i = tid & 31U; i < row4; i += 32U) {
        const float4 a = row[i];
        const float4 bqq = q[i];
        pacc += a.x * bqq.x + a.y * bqq.y + a.z * bqq.z + a.w * bqq.w;
      }
      double acc = q::WarpSumDouble(static_cast<double>(pacc));
      if ((tid & 31U) == 0U) {
        s_scores[g][base + t] = static_cast<float>(acc * scale);
      }
    }
    __syncthreads();
  }

  for (std::uint32_t g = 0; g < q::kDecodeGroup; ++g) {
    float local_max = q::kNegInf;
    for (std::uint32_t t = tid; t < tile_len; t += blockDim.x) {
      local_max = fmaxf(local_max, s_scores[g][t]);
    }
    const float tile_max = q::BlockMaxFloat(local_max, s_red[g]);
    float local_sum = 0.0F;
    for (std::uint32_t t = tid; t < tile_len; t += blockDim.x) {
      const float e = ::expf(s_scores[g][t] - tile_max);
      s_scores[g][t] = e;
      local_sum += e;
    }
    const float tile_sum = q::BlockSumFloat(local_sum, s_red[g]);
    if (tid == 0U) {
      s_meta[g][0] = tile_max;
      s_meta[g][1] = tile_sum;
    }
  }

  double acc[2][q::kDecodeGroup];
  for (std::uint32_t i = 0; i < 2U; ++i) {
#pragma unroll
    for (std::uint32_t g = 0; g < q::kDecodeGroup; ++g) {
      acc[i][g] = 0.0;
    }
  }
  fetch(0U, chunk_rows(0U), 0);
  // Pass B.
  for (std::uint32_t c = 0; c < chunks.x; ++c) {
    const std::uint32_t base = c * q::kDecodeRows;
    const std::uint32_t rows = chunk_rows(base);
    if (c + 1U < chunks.x) {
      fetch(base + q::kDecodeRows, chunk_rows(base + q::kDecodeRows),
            static_cast<int>(c + 1U) & 1);
    }
    store(rows, static_cast<int>(c) & 1);
    __syncthreads();
    const float* buf = &s_row[c & 1][0][0];
    for (std::uint32_t i = 0; i < 2U; ++i) {
      const std::uint32_t r = tid + i * blockDim.x;
      if (r >= kv_rank) {
        continue;
      }
      float pacc[q::kDecodeGroup];
#pragma unroll
      for (std::uint32_t g = 0; g < q::kDecodeGroup; ++g) {
        pacc[g] = 0.0F;
      }
      for (std::uint32_t t = 0; t < rows; ++t) {
        const float v = buf[t * q::kDecodeRowStride + r];
#pragma unroll
        for (std::uint32_t g = 0; g < q::kDecodeGroup; ++g) {
          pacc[g] += s_scores[g][base + t] * v;
        }
      }
#pragma unroll
      for (std::uint32_t g = 0; g < q::kDecodeGroup; ++g) {
        acc[i][g] += static_cast<double>(pacc[g]);
      }
    }
    __syncthreads();
  }
  for (std::uint32_t i = 0; i < 2U; ++i) {
    const std::uint32_t r = tid + i * blockDim.x;
    if (r >= kv_rank) {
      continue;
    }
#pragma unroll
    for (std::uint32_t g = 0; g < q::kDecodeGroup; ++g) {
      workspace[(static_cast<std::size_t>(tile_idx) * heads + h0 + g) *
                    (kv_rank + 2U) +
                2U + r] = static_cast<float>(acc[i][g]);
    }
  }
  if (tid < q::kDecodeGroup) {
    float* out =
        workspace + (static_cast<std::size_t>(tile_idx) * heads + h0 + tid) *
                        (kv_rank + 2U);
    out[0] = s_meta[tid][0];
    out[1] = s_meta[tid][1];
  }
}

constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kKvRank = 512;
constexpr std::uint32_t kRot = 64;
constexpr std::uint32_t kTile = 256;
constexpr std::uint32_t kThreads = 256;

void CheckCuda(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
    std::exit(1);
  }
}

float* Dev(std::size_t n) {
  float* p = nullptr;
  CheckCuda(hipMalloc(&p, n * sizeof(float)), "hipMalloc");
  return p;
}

void Oracle(const std::vector<float>& q_abs, const std::vector<float>& kv,
            const std::vector<float>& kpe, std::uint32_t n_pos, float scale,
            std::vector<float>* out) {
  out->assign(static_cast<std::size_t>(kHeads) * kKvRank, 0.0F);
  const std::size_t q_dim = kKvRank + kRot;
  for (std::uint32_t h = 0; h < kHeads; ++h) {
    const float* qa = q_abs.data() + static_cast<std::size_t>(h) * q_dim;
    std::vector<double> scores(n_pos);
    double max_score = -std::numeric_limits<double>::infinity();
    for (std::uint32_t p = 0; p < n_pos; ++p) {
      double acc = 0.0;
      for (std::uint32_t i = 0; i < kKvRank; ++i) {
        acc += static_cast<double>(qa[i]) *
               kv[static_cast<std::size_t>(p) * kKvRank + i];
      }
      for (std::uint32_t i = 0; i < kRot; ++i) {
        acc += static_cast<double>(qa[kKvRank + i]) *
               kpe[static_cast<std::size_t>(p) * kRot + i];
      }
      scores[p] = acc * static_cast<double>(scale);
      max_score = std::max(max_score, scores[p]);
    }
    double sum = 0.0;
    for (double& s : scores) {
      s = std::exp(s - max_score);
      sum += s;
    }
    float* dst = out->data() + static_cast<std::size_t>(h) * kKvRank;
    for (std::uint32_t r = 0; r < kKvRank; ++r) {
      double acc = 0.0;
      for (std::uint32_t p = 0; p < n_pos; ++p) {
        acc += (scores[p] / sum) *
               kv[static_cast<std::size_t>(p) * kKvRank + r];
      }
      dst[r] = static_cast<float>(acc);
    }
  }
}

double WorstRelative(const std::vector<float>& ref,
                     const std::vector<float>& got) {
  double worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double r = std::fabs(ref[i]);
    const double e = std::fabs(static_cast<double>(ref[i]) - got[i]);
    const double rel = r > 1e-6 ? e / r : e / 1e-6;
    if (rel > worst) worst = rel;
  }
  return worst;
}

struct Buffers {
  float* q;
  float* kv;
  float* kpe;
  float* ws;
  float* out;
  std::size_t ws_bytes;
};

Buffers MakeBuffers(std::uint32_t n_pos, const std::vector<float>& q_abs,
                    const std::vector<float>& kv,
                    const std::vector<float>& kpe) {
  Buffers b;
  b.q = Dev(q_abs.size());
  b.kv = Dev(kv.size());
  b.kpe = Dev(kpe.size());
  const std::size_t tiles = q::MlaDecodeTiles(n_pos, kTile);
  b.ws = Dev(tiles * kHeads * (kKvRank + 2U));
  b.out = Dev(static_cast<std::size_t>(kHeads) * kKvRank);
  CheckCuda(hipMemcpy(b.q, q_abs.data(), q_abs.size() * 4,
                      hipMemcpyHostToDevice),
            "copy q");
  CheckCuda(hipMemcpy(b.kv, kv.data(), kv.size() * 4, hipMemcpyHostToDevice),
            "copy kv");
  CheckCuda(hipMemcpy(b.kpe, kpe.data(), kpe.size() * 4,
                      hipMemcpyHostToDevice),
            "copy kpe");
  return b;
}

void RunVariant(int variant, const Buffers& b, std::uint32_t n_pos) {
  const std::uint32_t tiles = q::MlaDecodeTiles(n_pos, kTile);
  const float scale = 1.0F / std::sqrt(192.0F);
  if (variant == 2) {
    GroupDoubleBufKernel<<<dim3(kHeads / q::kDecodeGroup, tiles), kThreads>>>(
        b.q, b.kv, b.kpe, n_pos, kTile, scale, b.ws, kHeads, kKvRank, kRot);
  } else if (variant == 1) {
    q::MlaDecodePartialGroupKernel<<<dim3(kHeads / q::kDecodeGroup, tiles),
                                     kThreads>>>(
        b.q, b.kv, b.kpe, n_pos, kTile, scale, b.ws, kHeads, kKvRank, kRot);
  } else {
    q::MlaDecodePartialKernel<<<dim3(tiles, kHeads), kThreads>>>(
        b.q, b.kv, b.kpe, n_pos, kTile, scale, b.ws, kHeads, kKvRank, kRot);
  }
  q::MlaDecodeCombineKernel<<<kHeads, kThreads>>>(b.ws, tiles, kHeads, kKvRank,
                                                  b.out);
}

double TimeRun(int variant, const Buffers& b, std::uint32_t n_pos, int reps) {
  hipEvent_t e0, e1;
  CheckCuda(hipEventCreate(&e0), "event");
  CheckCuda(hipEventCreate(&e1), "event");
  RunVariant(variant, b, n_pos);
  CheckCuda(hipDeviceSynchronize(), "warm");
  CheckCuda(hipEventRecord(e0), "rec0");
  for (int i = 0; i < reps; ++i) {
    RunVariant(variant, b, n_pos);
  }
  CheckCuda(hipEventRecord(e1), "rec1");
  CheckCuda(hipEventSynchronize(e1), "sync");
  float ms = 0.0F;
  CheckCuda(hipEventElapsedTime(&ms, e0, e1), "elapsed");
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  return static_cast<double>(ms) * 1000.0 / reps;
}

std::vector<float> Random(std::size_t n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> v(n);
  for (float& x : v) x = dist(rng);
  return v;
}

}  // namespace

int main() {
  bool ok = true;
  std::printf("correctness:\n");
  for (std::uint32_t n : {std::uint32_t{7}, std::uint32_t{16},
                          std::uint32_t{33}, std::uint32_t{1025},
                          std::uint32_t{2050}}) {
    const std::size_t q_dim = kKvRank + kRot;
    auto q_abs = Random(kHeads * q_dim, 11 + n);
    auto kv = Random(static_cast<std::size_t>(n) * kKvRank, 21 + n);
    auto kpe = Random(static_cast<std::size_t>(n) * kRot, 31 + n);
    std::vector<float> ref, fused_out;
    Oracle(q_abs, kv, kpe, n, 1.0F / std::sqrt(192.0F), &ref);
    for (int variant : {0, 1, 2}) {
      Buffers b = MakeBuffers(n, q_abs, kv, kpe);
      RunVariant(variant, b, n);
      CheckCuda(hipDeviceSynchronize(), "run");
      std::vector<float> got(ref.size());
      CheckCuda(hipMemcpy(got.data(), b.out, got.size() * 4,
                          hipMemcpyDeviceToHost),
                "out");
      const double err = WorstRelative(ref, got);
      if (variant == 2) {
        const bool bit = got == fused_out && err <= 1e-4;
        ok = ok && bit;
        std::printf("  n=%5u doubuf worst %.3e bits=%s %s\n", n, err,
                    got == fused_out ? "yes" : "NO", bit ? "OK" : "FAIL");
      } else {
        std::printf("  n=%5u %-5s worst %.3e\n", n, variant ? "fused" : "old",
                    err);
        if (variant == 1) fused_out = got;
      }
      hipFree(b.q);
      hipFree(b.kv);
      hipFree(b.kpe);
      hipFree(b.ws);
      hipFree(b.out);
    }
  }
  std::printf("timing (us per MLA decode call, all 32 heads):\n");
  for (std::uint32_t n : {std::uint32_t{1024}, std::uint32_t{3072},
                          std::uint32_t{9216}, std::uint32_t{17408}}) {
    const std::size_t q_dim = kKvRank + kRot;
    auto q_abs = Random(kHeads * q_dim, 77 + n);
    auto kv = Random(static_cast<std::size_t>(n) * kKvRank, 88 + n);
    auto kpe = Random(static_cast<std::size_t>(n) * kRot, 99 + n);
    Buffers b = MakeBuffers(n, q_abs, kv, kpe);
    const double old_us = TimeRun(0, b, n, 20);
    const double new_us = TimeRun(1, b, n, 200);
    const double dbl_us = TimeRun(2, b, n, 200);
    std::printf("  n=%6u old %10.1f fused %9.1f (%4.2fx) doubuf %9.1f (%4.2fx vs fused)\n",
                n, old_us, new_us, old_us / new_us, dbl_us, new_us / dbl_us);
    hipFree(b.q);
    hipFree(b.kv);
    hipFree(b.kpe);
    hipFree(b.ws);
    hipFree(b.out);
  }
  return ok ? 0 : 1;
}
