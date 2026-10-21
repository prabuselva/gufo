// Nemotron Bolt decode router top-k bench (gfx1151): the production
// RouterKernel computes sigmoid+bias across the block, then thread 0 serially
// scans all n_experts LDS entries top_k times (tie-to-lowest-index,
// -FLT_MAX-skipping selection), which the decode profile put at ~135 us
// full-clock per launch over 40 layers. Probe a parallel (value, index)
// warp/block reduction per round that reproduces the oracle selection
// bitwise: strict-greater folds with the lower lane/index winning ties,
// invalid candidates never win, and the mask clears the selected index only
// (never equal-valued peers). Sigmoid values cached in LDS are the bit-exact
// same expf expression the serial path recomputes.
//
//   tools/bench/build.sh nem_decode_router_topk_bench && \
//   /tmp/nem_decode_router_topk_bench

#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

constexpr int kBlock = 512;
constexpr int kMaxExperts = 512;
constexpr int kNWarp = kBlock / 32;  // gfx1151 warpSize = 32

#define CHECK(expr)                                                    \
  do {                                                                 \
    const hipError_t err_ = (expr);                                    \
    if (err_ != hipSuccess) {                                          \
      std::fprintf(stderr, "hip error %s at %s:%d\n",                  \
                   hipGetErrorString(err_), __FILE__, __LINE__);       \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

// Bit-identical copy of production RouterKernel.
__global__ void RouterSerial(const float* __restrict__ logits,
                             const float* __restrict__ bias, int n_experts,
                             int top_k, int norm, float scale,
                             int32_t* __restrict__ ids,
                             float* __restrict__ weights) {
  const int token = blockIdx.x;
  const float* lg = logits + static_cast<int64_t>(token) * n_experts;
  __shared__ float scores[kMaxExperts];
  __shared__ int sel[16];
  __shared__ float num[16];
  for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
    scores[e] = 1.0F / (1.0F + expf(-lg[e])) + bias[e];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int k = 0; k < top_k; ++k) {
      int best = -1;
      for (int e = 0; e < n_experts; ++e) {
        const float v = scores[e];
        if (v == -FLT_MAX) {
          continue;
        }
        if (best < 0 || v > scores[best]) {
          best = e;
        }
      }
      sel[k] = best;
      if (best < 0) {
        break;
      }
      total = k + 1;
      scores[best] = -FLT_MAX;
      num[k] = 1.0F / (1.0F + expf(-lg[best]));
    }
    float sum = 0.0F;
    for (int k = 0; k < total; ++k) {
      sum += num[k];
    }
    if (norm) {
      sum = sum < 6.103515625e-5F ? 6.103515625e-5F : sum;
    }
    const float inv = sum > 0.0F ? 1.0F / sum : 0.0F;
    for (int k = 0; k < top_k; ++k) {
      const bool chosen = k < total;
      ids[static_cast<int64_t>(token) * top_k + k] =
          chosen ? sel[k] : static_cast<int32_t>(-1);
      weights[static_cast<int64_t>(token) * top_k + k] =
          chosen ? num[k] * inv * scale : 0.0F;
    }
  }
}

__device__ inline void fold_pair(float* val, int* idx, float oval, int oidx) {
  // Higher value wins; ties keep the already-held (lower-index) candidate;
  // invalid candidates (idx < 0) never win. Bitwise-strict '>' matches the
  // serial ascending scan whose first maximal element keeps the lowest index.
  const bool keep = *idx >= 0 && (oidx < 0 || *val > oval);
  *val = keep ? *val : oval;
  *idx = keep ? *idx : oidx;
}

// Parallel selection rounds over per-thread (score, index) candidates.
__global__ void RouterPar(const float* __restrict__ logits,
                          const float* __restrict__ bias, int n_experts,
                          int top_k, int norm, float scale,
                          int32_t* __restrict__ ids,
                          float* __restrict__ weights) {
  const int token = blockIdx.x;
  const float* lg = logits + static_cast<int64_t>(token) * n_experts;
  __shared__ float sigv[kMaxExperts];
  __shared__ float wval[kNWarp];
  __shared__ int widx[kNWarp];
  __shared__ float num[16];
  __shared__ int sel[16];
  __shared__ int gidx;
  __shared__ int stop_k;

  const int tid = static_cast<int>(threadIdx.x);
  __shared__ float score[kMaxExperts];
  if (tid < n_experts) {
    const float sig = 1.0F / (1.0F + expf(-lg[tid]));
    sigv[tid] = sig;
    score[tid] = sig + bias[tid];
  }
  bool alive = tid < n_experts && score[tid] != -FLT_MAX;
  if (tid == 0) {
    stop_k = -1;
  }
  __syncthreads();

  for (int k = 0; k < top_k; ++k) {
    // Fresh per-round candidate: the fold below clobbers registers, so the
    // per-round source is the immutable shared score + per-thread alive bit.
    float val = alive ? score[tid] : -FLT_MAX;
    int idx = alive ? tid : -1;
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const float oval = __shfl_down_sync(~0ULL, val, off);
      const int oidx = __shfl_down_sync(~0ULL, idx, off);
      fold_pair(&val, &idx, oval, oidx);
    }
    if ((tid & (warpSize - 1)) == 0) {
      wval[tid >> 5] = val;
      widx[tid >> 5] = idx;
    }
    __syncthreads();
    if (tid < warpSize) {
      float mval = tid < kNWarp ? wval[tid] : -FLT_MAX;
      int midx = tid < kNWarp ? widx[tid] : -1;
      for (int off = warpSize / 2; off > 0; off >>= 1) {
        const float oval = __shfl_down_sync(~0ULL, mval, off);
        const int oidx = __shfl_down_sync(~0ULL, midx, off);
        fold_pair(&mval, &midx, oval, oidx);
      }
      if (tid == 0) {
        gidx = midx;
        if (midx >= 0) {
          sel[k] = midx;
          num[k] = sigv[midx];
        } else {
          stop_k = k;
        }
      }
    }
    __syncthreads();
    if (gidx < 0) {
      break;
    }
    if (tid == gidx) {
      alive = false;  // selected index masked; equal-valued peers stay eligible
    }
  }

  if (tid == 0) {
    const int total = stop_k >= 0 ? stop_k : top_k;
    float sum = 0.0F;
    for (int k = 0; k < total; ++k) {
      sum += num[k];
    }
    if (norm) {
      sum = sum < 6.103515625e-5F ? 6.103515625e-5F : sum;
    }
    const float inv = sum > 0.0F ? 1.0F / sum : 0.0F;
    for (int k = 0; k < top_k; ++k) {
      const bool chosen = k < total;
      ids[static_cast<int64_t>(token) * top_k + k] =
          chosen ? sel[k] : static_cast<int32_t>(-1);
      weights[static_cast<int64_t>(token) * top_k + k] =
          chosen ? num[k] * inv * scale : 0.0F;
    }
  }
}

int main() {
  const int top_k = 8;
  const int cases[][3] = {
      // n_experts, ties (many equal logits), mask_tail (last m pre-FLT_MAX)
      {240, 1, 0}, {368, 1, 0}, {240, 0, 0}, {368, 0, 0}, {240, 1, 8},
  };
  const int iters = 4000;
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  for (const auto& c : cases) {
    const int ne = c[0];
    std::vector<float> hlg(ne), hbias(ne);
    uint32_t st = 1234567u + static_cast<uint32_t>(ne * 31 + c[1] * 7 + c[2]);
    for (int i = 0; i < ne; ++i) {
      st = st * 1664525u + 1013904223u;
      hlg[i] = (static_cast<float>(st >> 8) / 8388608.0F) * 2.0F - 1.0F;
      if (c[1] && i % 5 == 0) {
        hlg[i] = 0.75F;  // force exact ties incl. top region
      }
      st = st * 1664525u + 1013904223u;
      hbias[i] = (static_cast<float>(st >> 8) / 8388608.0F) * 0.25F
                 - 0.125F;
    }
    float *d_lg, *d_bias, *d_w0, *d_w1;
    int32_t *d_id0, *d_id1;
    const size_t fbytes = sizeof(float) * ne;
    CHECK(hipMalloc(&d_lg, fbytes));
    CHECK(hipMalloc(&d_bias, fbytes));
    CHECK(hipMalloc(&d_w0, sizeof(float) * top_k));
    CHECK(hipMalloc(&d_w1, sizeof(float) * top_k));
    CHECK(hipMalloc(&d_id0, sizeof(int32_t) * top_k));
    CHECK(hipMalloc(&d_id1, sizeof(int32_t) * top_k));
    CHECK(hipMemcpy(d_lg, hlg.data(), fbytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_bias, hbias.data(), fbytes, hipMemcpyHostToDevice));
    int mism = 0;
    for (int norm = 0; norm <= 1; ++norm) {
      const int32_t sentinel = 0x5A5A5A5A;
      std::vector<int32_t> ids0(top_k, sentinel), ids1(top_k, sentinel);
      std::vector<float> w0(top_k, -1.0F), w1(top_k, -1.0F);
      CHECK(hipMemset(d_id0, 0x5A, sizeof(int32_t) * top_k));
      CHECK(hipMemset(d_id1, 0x5A, sizeof(int32_t) * top_k));
      RouterSerial<<<1, kBlock>>>(d_lg, d_bias, ne, top_k, norm, 1.0F, d_id0,
                                  d_w0);
      RouterPar<<<1, kBlock>>>(d_lg, d_bias, ne, top_k, norm, 1.0F, d_id1,
                               d_w1);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(ids0.data(), d_id0, sizeof(int32_t) * top_k,
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(ids1.data(), d_id1, sizeof(int32_t) * top_k,
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(w0.data(), d_w0, sizeof(float) * top_k,
                      hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(w1.data(), d_w1, sizeof(float) * top_k,
                      hipMemcpyDeviceToHost));
      if (std::memcmp(ids0.data(), ids1.data(), sizeof(int32_t) * top_k) != 0 ||
          std::memcmp(w0.data(), w1.data(), sizeof(float) * top_k) != 0) {
        ++mism;
        std::printf("  MISMATCH ne=%d norm=%d\n", ne, norm);
        for (int k = 0; k < top_k; ++k) {
          std::printf("    k=%d serial %d %.9g | par %d %.9g\n", k, ids0[k],
                      w0[k], ids1[k], w1[k]);
        }
      }
    }
    float ms_s = 0.0F, ms_p = 0.0F, dummy = 0.0F;
    for (int pass = 0; pass < 2; ++pass) {
      CHECK(hipEventRecord(e0));
      for (int i = 0; i < iters; ++i) {
        RouterSerial<<<1, kBlock>>>(d_lg, d_bias, ne, top_k, 1, 1.0F, d_id0,
                                    d_w0);
      }
      CHECK(hipEventRecord(e1));
      CHECK(hipEventSynchronize(e1));
      if (pass) {
        CHECK(hipEventElapsedTime(&ms_s, e0, e1));
      }
      CHECK(hipEventRecord(e0));
      for (int i = 0; i < iters; ++i) {
        RouterPar<<<1, kBlock>>>(d_lg, d_bias, ne, top_k, 1, 1.0F, d_id1,
                                 d_w1);
      }
      CHECK(hipEventRecord(e1));
      CHECK(hipEventSynchronize(e1));
      if (pass) {
        CHECK(hipEventElapsedTime(&ms_p, e0, e1));
      }
    }
    std::printf(
        "  ne=%d ties=%d: bitwise %s  serial %7.2f us  par %7.2f us  "
        "x%.2f\n",
        ne, c[1], mism ? "FAIL" : "OK", ms_s * 1000.0F / iters,
        ms_p * 1000.0F / iters, ms_s / ms_p);
    (void)dummy;
    hipFree(d_lg);
    hipFree(d_bias);
    hipFree(d_w0);
    hipFree(d_w1);
    hipFree(d_id0);
    hipFree(d_id1);
  }
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  return 0;
}
