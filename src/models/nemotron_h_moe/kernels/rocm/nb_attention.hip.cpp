#include "src/models/nemotron_h_moe/kernels/rocm/nb_attention.h"

#include <cstdint>

namespace {

constexpr int kDim = 128;  // max head dimension; one thread per dimension.
constexpr float kNegInf = -3.4e38F;

__global__ void AttnAppendKernel(const float* __restrict__ k_src,
                                 const float* __restrict__ v_src,
                                 float* __restrict__ kv, int pos0, int tokens,
                                 int kvdim) {
  const int64_t n = static_cast<int64_t>(tokens) * (2 * kvdim);
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int pos = pos0 + static_cast<int>(i / (2 * kvdim));
    const int64_t slot = i % (2 * kvdim);
    const int64_t t = (i - slot) / (2 * kvdim);
    const float* src = slot < kvdim ? k_src + t * kvdim + slot
                                    : v_src + t * kvdim + (slot - kvdim);
    kv[static_cast<int64_t>(pos) * (2 * kvdim) + slot] = *src;
  }
}

// Flash-style online-softmax causal attention, one block per (query head,
// row tile). threads are dimension lanes; o[r] accumulates output element
// `tid` of tile row r across the growing context.
// K rows live in LDS transposed (skT[d][j + 1]; the +1 pad avoids the 32-way
// bank conflict seen with sk[j][d] when j is lane-distributed in the score
// loop). Measured 3.2x at pp16384 shapes with TR=24, kTile=16 (gfx1151).
template<int TR, int kTile>
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
  __shared__ float skT[kDim][kTile + 1];
  __shared__ float sv[kTile][kDim];
  __shared__ float se[TR][kTile];
  __shared__ float sm[TR];
  __shared__ float sl[TR];
  __shared__ float sa[TR];

  for (int i = tid; i < TR * kDim; i += kDim) {
    const int r = i / kDim;
    const int d = i % kDim;
    sq[r][d] = r < rows && d < hd
                   ? q[static_cast<int64_t>(row0 + r) * heads * hd + h * hd + d]
                   : 0.0F;
  }
  if (tid < TR) {
    sm[tid] = kNegInf;
    sl[tid] = 0.0F;
  }
  __syncthreads();

  float o[TR] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  const int64_t kvrow = 2 * kvdim;
  const float* kv_k = kv + g * hd;
  const float* kv_v = kv + kvdim + g * hdv;
  const int nmax = pos0 + row0 + rows;  // exclusive causal bound for this tile

  for (int p0 = 0; p0 < nmax; p0 += kTile) {
    const int nj = nmax - p0 < kTile ? nmax - p0 : kTile;
    for (int i = tid; i < nj * kDim; i += kDim) {
      const int j = i / kDim;
      const int d = i % kDim;
      skT[d][j + 1] =
          d < hd ? kv_k[(static_cast<int64_t>(p0 + j)) * kvrow + d] : 0.0F;
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
        for (int d = 0; d < kDim; ++d) {
          acc += sq[r][d] * skT[d][j + 1];
        }
        se[r][j] = acc * scale;
      } else {
        se[r][j] = kNegInf;
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
}

}  // namespace

extern "C" int nbl_attn_append(const float* k_src, const float* v_src,
                               float* kv, int pos0, int tokens, int kvdim,
                               hipStream_t stream) {
  if (!k_src || !v_src || !kv || pos0 < 0 || tokens <= 0 || kvdim <= 0) {
    return -1;
  }
  const int64_t n = static_cast<int64_t>(tokens) * (2 * kvdim);
  const int grid = static_cast<int>((n + 255) / 256);
  const int launch = grid < 1024 ? grid : 1024;
  AttnAppendKernel<<<launch, 256, 0, stream>>>(k_src, v_src, kv, pos0, tokens,
                                               kvdim);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

extern "C" int nbl_attn(const float* q, const float* kv, int pos0, int tokens,
                        float* out, int heads, int hd, int hdv, int kvh,
                        int kvdim, float scale, hipStream_t stream) {
  if (!q || !kv || !out || pos0 < 0 || tokens <= 0 || heads <= 0 || kvh <= 0 ||
      heads % kvh != 0 || hd <= 0 || hd > kDim || hdv <= 0 || hdv > kDim ||
      kvdim < kvh * hd) {
    return -1;
  }
  // Partial tiles (including decode with tokens == 1) take the guarded path.
  const dim3 grid(static_cast<unsigned>((tokens + 23) / 24),
                  static_cast<unsigned>(heads));
  AttnKernel<24, 16><<<grid, kDim, 0, stream>>>(q, kv, pos0, tokens, out, heads,
                                                hd, hdv, kvh, kvdim, scale);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}
