// Nemotron Bolt NoPE attention prefill microbench (gfx1151).
//
// Production shape: q [tokens, heads=32, hd=128] f32, kv rows
// [tokens][2*kvdim] with kvdim = kvh*hd = 256 (K0 K1 V0 V1), kvh=2, causal,
// scale 1/sqrt(128). One block per (query head, row tile), 128 threads.
// Baseline = production AttnKernel<8>; variants probe the score-phase LDS
// bank conflict (sk(j,d) with j lane-distributed) and the token tile width.
//
// tools/bench/build.sh nem_attn_bench && /tmp/nem_attn_bench [L...]

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

constexpr int kDim = 128;
constexpr float kNeg = -3.4e38F;
constexpr int kHeads = 32;
constexpr int kKvh = 2;
constexpr int kHd = 128;
constexpr int kHdv = 128;
constexpr int kKvdim = kKvh * kHd;  // 256 float per row per half
constexpr float kScale = 0.08838834764831845F;

// ---------------- variant kernel (production copy + skT/TR knobs) ----------
template <int TR, int kTile, int SKT>
__global__ void AttnKernel(const float* __restrict__ q,
                           const float* __restrict__ kv, int pos0, int tokens,
                           float* __restrict__ out, int heads, int hd, int hdv,
                           int kvh, int kvdim, float scale) {
  const int h = static_cast<int>(blockIdx.y);
  const int row0 = static_cast<int>(blockIdx.x) * TR;
  const int rows = tokens - row0 < TR ? tokens - row0 : TR;
  const int g = h * kvh / heads;
  const int tid = static_cast<int>(threadIdx.x);

  __shared__ float sq[TR][kDim];
  __shared__ float skb[SKT ? 1 : kTile * kDim];
  __shared__ float skTb[SKT ? kDim * (kTile + 1) : 1];
  __shared__ float sv[kTile][kDim];
  __shared__ float se[TR][kTile];
  __shared__ float sm[TR];
  __shared__ float sl[TR];
  __shared__ float sa[TR];
#define sk(j, d) skb[(j) * kDim + (d)]
#define skT(d, j) skTb[(d) * (kTile + 1) + (j)]

  for (int i = tid; i < TR * kDim; i += kDim) {
    const int r = i / kDim;
    const int d = i % kDim;
    sq[r][d] = r < rows && d < hd
                   ? q[static_cast<int64_t>(row0 + r) * heads * hd + h * hd + d]
                   : 0.0F;
  }
  if (tid < TR) {
    sm[tid] = kNeg;
    sl[tid] = 0.0F;
  }
  __syncthreads();

  float o[TR];
#pragma unroll 1
  for (int r = 0; r < TR; ++r) o[r] = 0.0F;
  const int64_t kvrow = 2 * kvdim;
  const float* kv_k = kv + g * hd;
  const float* kv_v = kv + kvdim + g * hdv;
  const int nmax = pos0 + row0 + rows;

  for (int p0 = 0; p0 < nmax; p0 += kTile) {
    const int nj = nmax - p0 < kTile ? nmax - p0 : kTile;
    if (SKT) {
      for (int i = tid; i < nj * kDim; i += kDim) {
        const int j = i / kDim;
        const int d = i % kDim;
        skT(d, j) = kv_k[(static_cast<int64_t>(p0 + j)) * kvrow + d];
      }
    } else {
      for (int i = tid; i < nj * kDim; i += kDim) {
        const int j = i / kDim;
        const int d = i % kDim;
        sk(j, d) =
            d < hd ? kv_k[(static_cast<int64_t>(p0 + j)) * kvrow + d] : 0.0F;
      }
    }
    for (int i = tid; i < nj * kDim; i += kDim) {
      const int j = i / kDim;
      const int d = i % kDim;
      sv[j][d] =
          d < hdv ? kv_v[(static_cast<int64_t>(p0 + j)) * kvrow + d] : 0.0F;
    }
    __syncthreads();

    for (int i = tid; i < TR * kTile; i += kDim) {
      const int r = i / kTile;
      const int j = i % kTile;
      const int p = p0 + j;
      if (r < rows && p <= pos0 + row0 + r && p < nmax) {
        float acc = 0.0F;
        if (SKT) {
          for (int d = 0; d < kDim; ++d) acc += sq[r][d] * skT(d, j);
        } else {
          for (int d = 0; d < kDim; ++d) acc += sq[r][d] * sk(j, d);
        }
        se[r][j] = acc * scale;
      } else {
        se[r][j] = kNeg;
      }
    }
    __syncthreads();

    if (tid < TR) {
      float mnew = sm[tid];
      for (int j = 0; j < kTile; ++j) {
        mnew = se[tid][j] > mnew ? se[tid][j] : mnew;
      }
      sa[tid] = expf(sm[tid] - mnew);
      sm[tid] = mnew;
    }
    __syncthreads();

    for (int i = tid; i < TR * kTile; i += kDim) {
      const int r = i / kTile;
      const int j = i % kTile;
      se[r][j] = expf(se[r][j] - sm[r]);
    }
    __syncthreads();

    if (tid < TR) {
      float ssum = 0.0F;
      for (int j = 0; j < kTile; ++j) {
        ssum += se[tid][j];
      }
      sl[tid] = sl[tid] * sa[tid] + ssum;
    }

    for (int r = 0; r < TR; ++r) {
      o[r] *= sa[r];
    }
    for (int j = 0; j < nj; ++j) {
      for (int r = 0; r < TR; ++r) {
        o[r] += se[r][j] * sv[j][tid];
      }
    }
    __syncthreads();
  }

  if (tid < hdv) {
    for (int r = 0; r < rows; ++r) {
      out[static_cast<int64_t>(row0 + r) * heads * hdv + h * hdv + tid] =
          o[r] / sl[r];
    }
  }
#undef sk
#undef skT
}

template <int TR, int kTile, int SKT>
void launch(const float* q, const float* kv, int tokens, float* out,
            hipStream_t s) {
  const dim3 grid(static_cast<unsigned>((tokens + TR - 1) / TR), kHeads);
  AttnKernel<TR, kTile, SKT><<<grid, kDim, 0, s>>>(
      q, kv, 0, tokens, out, kHeads, kHd, kHdv, kKvh, kKvdim, kScale);
}

// ------------- CPU f64 reference (causal GQA SDPA) -------------
std::vector<double> cpu_attn(const std::vector<float>& q,
                             const std::vector<float>& kv, int tokens) {
  const int64_t kvrow = 2 * kKvdim;
  std::vector<double> out(static_cast<int64_t>(tokens) * kHeads * kHdv, 0.0);
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < kHeads; ++h) {
      const int g = h * kKvh / kHeads;
      std::vector<double> e(t + 1);
      double mx = -1e300;
      for (int j = 0; j <= t; ++j) {
        double acc = 0.0;
        for (int d = 0; d < kHd; ++d) {
          acc += static_cast<double>(
                     q[static_cast<int64_t>(t) * kHeads * kHd + h * kHd + d]) *
                 static_cast<double>(kv[j * kvrow + g * kHd + d]);
        }
        e[j] = acc * kScale;
        mx = e[j] > mx ? e[j] : mx;
      }
      double s = 0.0;
      for (int j = 0; j <= t; ++j) s += std::exp(e[j] - mx);
      for (int d = 0; d < kHdv; ++d) {
        double acc = 0.0;
        for (int j = 0; j <= t; ++j) {
          acc += std::exp(e[j] - mx) *
                 static_cast<double>(
                     kv[j * kvrow + kKvdim + g * kHdv + d]);
        }
        out[static_cast<int64_t>(t) * kHeads * kHdv + h * kHdv + d] = acc / s;
      }
    }
  }
  return out;
}

float* dupload(const std::vector<float>& h) {
  float* p = nullptr;
  CHECK(hipMalloc(&p, h.size() * sizeof(float)));
  CHECK(hipMemcpy(p, h.data(), h.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  return p;
}

struct Res {
  double ms;
  double ts;
};

template <typename F>
Res timeit(F&& fn, int reps) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  fn();  // warm
  CHECK(hipDeviceSynchronize());
  CHECK(hipEventRecord(a));
  for (int i = 0; i < reps; ++i) fn();
  CHECK(hipEventRecord(b));
  CHECK(hipEventSynchronize(b));
  float ms = 0.0F;
  CHECK(hipEventElapsedTime(&ms, a, b));
  CHECK(hipEventDestroy(a));
  CHECK(hipEventDestroy(b));
  return {ms / reps, 0.0};
}

}  // namespace

int main(int argc, char** argv) {
  int check_len = 131;
  std::vector<int> lens;
  for (int i = 1; i < argc; ++i) lens.push_back(std::atoi(argv[i]));
  if (lens.empty()) lens = {2048, 8192, 16384};

  // Correctness check vs f64 reference at check_len for all SKT/TR shapes.
  {
    std::vector<float> q(static_cast<size_t>(check_len) * kHeads * kHd), kv(
        static_cast<size_t>(check_len) * 2 * kKvdim);
    uint32_t st = 12345;
    auto rnd = [&st]() {
      st = st * 1664525U + 1013904223U;
      return static_cast<float>(static_cast<int>(st >> 8) % 20000 - 10000) /
             10000.0F;
    };
    for (auto& x : q) x = rnd();
    for (auto& x : kv) x = rnd();
    auto ref = cpu_attn(q, kv, check_len);
    float* dq = dupload(q);
    float* dkv = dupload(kv);
    std::vector<float> cand(static_cast<size_t>(check_len) * kHeads * kHdv);
    float* dcand = nullptr;
    CHECK(hipMalloc(&dcand, cand.size() * sizeof(float)));
    struct V {
      const char* name;
      void (*run)(const float*, const float*, int, float*, hipStream_t);
    };
    const V variants[] = {
      {"A tr8 kt32        ", launch<8, 32, 0>},
      {"G tr24 kt16 skT   ", launch<24, 16, 1>},
      {"H tr20 kt16 skT   ", launch<20, 16, 1>},
      {"I tr28 kt16 skT   ", launch<28, 16, 1>},
      };
    double worst = 0.0;
    for (auto& v : variants) {
      CHECK(hipMemset(dcand, 0, cand.size() * sizeof(float)));
      v.run(dq, dkv, check_len, dcand, nullptr);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(cand.data(), dcand, cand.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      double num = 0.0, den = 0.0;
      size_t nan_ref = 0, nan_cand = 0;
      for (size_t i = 0; i < cand.size(); ++i) {
        if (std::isnan(ref[i])) ++nan_ref;
        if (std::isnan(cand[i])) ++nan_cand;
        if (std::isnan(cand[i]) || std::isnan(ref[i])) {
          std::printf("NaN at %zu: cand=%f ref=%f\n", i, cand[i], ref[i]);
          return 1;
        }
        const double d = static_cast<double>(cand[i]) - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
      }
      const double max_rel = std::sqrt(num / den);
      if (max_rel > worst) worst = max_rel;
      std::printf(
          "  %s max_rel=%.3e nan_ref=%zu nan_cand=%zu "
          "ref0=%.6f cand0=%.6f %s\n",
          v.name, max_rel, nan_ref, nan_cand, ref[0], cand[0],
          max_rel < 2e-4 ? "OK" : "FAIL");
    }
    if (worst >= 2e-4) {
      std::printf("correctness FAILED\n");
      return 1;
    }
    // check_len also exercises the ragged tail (131 = 8 tiles + 19 rows).
  }

  float* dq;
  float* dkv;
  float* dout;
  int maxlen = 0;
  for (int L : lens) maxlen = L > maxlen ? L : maxlen;
  CHECK(hipMalloc(&dq, static_cast<size_t>(maxlen) * kHeads * kHd * 4));
  CHECK(hipMalloc(&dkv, static_cast<size_t>(maxlen) * 2 * kKvdim * 4));
  CHECK(hipMalloc(&dout, static_cast<size_t>(maxlen) * kHeads * kHdv * 4));
  CHECK(hipMemset(dq, 0x3d, static_cast<size_t>(maxlen) * kHeads * kHd * 4));
  CHECK(hipMemset(dkv, 0x3d, static_cast<size_t>(maxlen) * 2 * kKvdim * 4));

  std::printf("\n%-16s", "variant");
  for (int L : lens) std::printf("%10s", "t/s");
  std::printf("\n");

  struct V {
    const char* name;
    void (*run)(const float*, const float*, int, float*, hipStream_t);
  };
  const V variants[] = {
      {"A tr8 kt32        ", launch<8, 32, 0>},
      {"G tr24 kt16 skT   ", launch<24, 16, 1>},
      {"H tr20 kt16 skT   ", launch<20, 16, 1>},
      {"I tr28 kt16 skT   ", launch<28, 16, 1>},
  };
  for (auto& v : variants) {
    std::printf("%-16s", v.name);
    for (int L : lens) {
      int reps = L >= 16384 ? 3 : 10;
      auto r = timeit([&] { v.run(dq, dkv, L, dout, nullptr); }, reps);
      std::printf("%10.1f", L / (r.ms * 1e-3));
    }
    std::printf("\n");
  }

  CHECK(hipFree(dq));
  CHECK(hipFree(dkv));
  CHECK(hipFree(dout));
  return 0;
}
