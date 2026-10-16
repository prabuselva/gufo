// Standalone A/B for the Ling3 MLA prefill: staged-scalar kernel vs the
// fused register-resident online-softmax kernel, both checked against the
// fp64 host oracle.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_fp16.h>

#include "src/models/ling3_flash/kernels/rocm/attention.hip.cpp"

namespace q = gufo::models::ling3_flash::rocm;

namespace {

constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kKvRank = 512;
constexpr std::uint32_t kRot = 64;

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

// start_pos keys plus `tokens` new queries with causal masking; the oracle
// mirrors the executor convention latent[token] attends keys <=
// start_pos + token.
void Oracle(const std::vector<float>& q_abs, const std::vector<float>& q_rope,
            const std::vector<float>& kv, const std::vector<float>& kpe,
            std::uint32_t start_pos, std::uint32_t tokens, float scale,
            std::vector<float>* out) {
  out->assign(static_cast<std::size_t>(tokens) * kHeads * kKvRank, 0.0F);
  for (std::uint32_t t = 0; t < tokens; ++t) {
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const float* qa = q_abs.data() +
                        (static_cast<std::size_t>(t) * kHeads + h) * kKvRank;
      const float* qr = q_rope.data() +
                        (static_cast<std::size_t>(t) * kHeads + h) * kRot;
      const std::uint32_t keys = start_pos + t + 1U;
      std::vector<double> scores(keys);
      double max_score = -std::numeric_limits<double>::infinity();
      for (std::uint32_t p = 0; p < keys; ++p) {
        double acc = 0.0;
        for (std::uint32_t i = 0; i < kKvRank; ++i) {
          acc += static_cast<double>(qa[i]) *
                 kv[static_cast<std::size_t>(p) * kKvRank + i];
        }
        for (std::uint32_t i = 0; i < kRot; ++i) {
          acc += static_cast<double>(qr[i]) *
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
      float* dst = out->data() +
                   (static_cast<std::size_t>(t) * kHeads + h) * kKvRank;
      for (std::uint32_t r = 0; r < kKvRank; ++r) {
        double acc = 0.0;
        for (std::uint32_t p = 0; p < keys; ++p) {
          acc += (scores[p] / sum) *
                 kv[static_cast<std::size_t>(p) * kKvRank + r];
        }
        dst[r] = static_cast<float>(acc);
      }
    }
  }
}

struct ErrStat {
  double worst = 0.0;
  std::size_t nan = 0;
  std::size_t worst_idx = 0;
};

ErrStat WorstRelative(const std::vector<float>& ref,
                      const std::vector<float>& got) {
  ErrStat s;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double r = std::fabs(ref[i]);
    const double d = static_cast<double>(ref[i]) - got[i];
    const double e = std::fabs(d);
    if (std::isnan(e) || std::isnan(ref[i]) || std::isnan(got[i])) {
      ++s.nan;
      continue;
    }
    const double rel = r > 1e-3 ? e / r : e / 1e-3;
    if (rel > s.worst) {
      s.worst = rel;
      s.worst_idx = i;
    }
  }
  return s;
}

struct Buffers {
  float* q;
  float* qr;
  float* kv;
  float* kpe;
  float* out;
};

Buffers MakeBuffers(std::uint32_t rows, std::uint32_t tokens,
                    const std::vector<float>& q_abs,
                    const std::vector<float>& q_rope,
                    const std::vector<float>& kv,
                    const std::vector<float>& kpe) {
  Buffers b;
  b.q = Dev(q_abs.size());
  b.qr = Dev(q_rope.size());
  b.kv = Dev(kv.size());
  b.kpe = Dev(kpe.size());
  b.out = Dev(static_cast<std::size_t>(tokens) * kHeads * kKvRank);
  CheckCuda(hipMemcpy(b.q, q_abs.data(), q_abs.size() * 4,
                      hipMemcpyHostToDevice),
            "copy q");
  CheckCuda(hipMemcpy(b.qr, q_rope.data(), q_rope.size() * 4,
                      hipMemcpyHostToDevice),
            "copy qr");
  CheckCuda(hipMemcpy(b.kv, kv.data(), kv.size() * 4, hipMemcpyHostToDevice),
            "copy kv");
  CheckCuda(hipMemcpy(b.kpe, kpe.data(), kpe.size() * 4,
                      hipMemcpyHostToDevice),
            "copy kpe");
  (void)rows;
  return b;
}


void RunVariant(bool fused, const Buffers& b, std::uint32_t start_pos,
                std::uint32_t tokens);


// CANDIDATE: fused-prefill variant with the staged KV row reused in
// registers between the QK dot and the PV update. Operations and their
// order are bit-identical to MlaPrefillFusedKernel; only LDS traffic halves.
template<std::uint32_t KvSlice, std::uint32_t Warps, std::uint32_t WarpQ>
__global__ void FusedKVReuse(
    const float* __restrict__ q_abs, const float* __restrict__ q_rope,
    const float* __restrict__ kv, const float* __restrict__ kpe,
    float* __restrict__ latent, std::uint32_t start_pos, std::uint32_t tokens,
    std::uint32_t heads, std::uint32_t kv_rank, std::uint32_t rotary,
    float scale) {
  using namespace gufo::models::ling3_flash::rocm;
  namespace qn = gufo::models::ling3_flash::rocm;
  constexpr std::uint32_t kStride = qn::kPrefillMaxKv + qn::kPrefillMaxRot;
  constexpr std::uint32_t kKT = qn::kPrefillFusedKT;
  __shared__ float s_kv[kKT][kStride];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t q0 = blockIdx.y * (Warps * WarpQ);
  const std::uint32_t q_last = min(q0 + (Warps * WarpQ), tokens);
  const std::uint32_t row_dim = kv_rank + rotary;
  const std::uint32_t warp = tid >> 5U;
  const std::uint32_t wi0 = q0 + warp * WarpQ;
  const bool warp_active = wi0 < q_last;
  float qreg[WarpQ][KvSlice];
  float qrot[WarpQ][2U];
  float acc[WarpQ][KvSlice];
  float rmax[WarpQ];
  float rsum[WarpQ];
  for (std::uint32_t i = 0; i < WarpQ; ++i) {
    const std::uint32_t token = wi0 + i;
    const bool valid = token < q_last;
    const std::size_t qo =
        (static_cast<std::size_t>(valid ? token : q0) * heads + h) * kv_rank;
    const std::size_t ro =
        (static_cast<std::size_t>(valid ? token : q0) * heads + h) * rotary;
    for (std::uint32_t d = 0; d < KvSlice; ++d) {
      qreg[i][d] = valid ? q_abs[qo + lane + 32U * d] : 0.0F;
      acc[i][d] = 0.0F;
    }
    qrot[i][0U] = valid ? q_rope[ro + lane] : 0.0F;
    qrot[i][1U] = valid ? q_rope[ro + lane + 32U] : 0.0F;
    rmax[i] = qn::kNegInf;
    rsum[i] = 0.0F;
  }
  const std::uint32_t causal_max = start_pos + q_last;
  for (std::uint32_t base = 0; base < causal_max; base += kKT) {
    __syncthreads();
    for (std::uint32_t idx = tid; idx < kKT * row_dim; idx += blockDim.x) {
      const std::uint32_t k = idx / row_dim;
      const std::uint32_t d = idx % row_dim;
      const std::uint32_t j = base + k;
      s_kv[k][d] = (j < causal_max)
                       ? ((d < kv_rank)
                              ? kv[static_cast<std::size_t>(j) * kv_rank + d]
                              : kpe[static_cast<std::size_t>(j) * rotary +
                                    (d - kv_rank)])
                       : 0.0F;
    }
    __syncthreads();
    const std::uint32_t rows =
        causal_max - base < kKT ? causal_max - base : kKT;
    const std::uint32_t qend =
        wi0 + WarpQ < q_last ? wi0 + WarpQ : q_last;
    if (!warp_active || base >= start_pos + qend) {
      continue;
    }
#pragma unroll 4
    for (std::uint32_t k = 0; k < rows; ++k) {
      float v[KvSlice];
      for (std::uint32_t d = 0; d < KvSlice; ++d) {
        v[d] = s_kv[k][lane + 32U * d];
      }
      for (std::uint32_t i = 0; i < WarpQ; ++i) {
        const std::uint32_t token = wi0 + i;
        float pa = qrot[i][0U] * s_kv[k][kv_rank + lane];
        float pb = qrot[i][1U] * s_kv[k][kv_rank + lane + 32U];
        for (std::uint32_t d = 0; d < KvSlice / 2U; ++d) {
          pa += qreg[i][d] * v[d];
          pb += qreg[i][d + KvSlice / 2U] * v[d + KvSlice / 2U];
        }
        float p = pa + pb;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          p += __shfl_xor_sync(~0ULL, p, off);
        }
        float score = p * scale;
        if (token >= q_last || base + k > start_pos + token) {
          score = qn::kNegInf;
        }
        const float m = fmaxf(rmax[i], score);
        const float corr =
            (rmax[i] == qn::kNegInf) ? 0.0F : ::expf(rmax[i] - m);
        const float e = (score == qn::kNegInf) ? 0.0F : ::expf(score - m);
        rsum[i] = rsum[i] * corr + e;
        rmax[i] = m;
        for (std::uint32_t d = 0; d < KvSlice; ++d) {
          acc[i][d] = acc[i][d] * corr + e * v[d];
        }
      }
    }
  }
  for (std::uint32_t i = 0; i < WarpQ; ++i) {
    const std::uint32_t token = wi0 + i;
    if (token >= q_last) {
      continue;
    }
    const std::size_t lo =
        (static_cast<std::size_t>(token) * heads + h) * kv_rank;
    for (std::uint32_t d = 0; d < KvSlice; ++d) {
      latent[lo + lane + 32U * d] = acc[i][d] / rsum[i];
    }
  }
}


// ---------------------------------------------------------------------------
// CANDIDATE 4 (lever 5.3c): WMMA absorbed-MLA prefill in the gemma4 3.4a
// shape, adapted to Ling3 (kv_heads = 1: every head attends the same cached
// latent row, so 2 heads x 16 queries form one row block) and to the 5.1a
// numerical contract (QK = Qh*Kh + Qh*Kl + Ql*Kh, fp32 accumulate; the
// rejected 5.1b route died on a separate scattered V-cache refetch - here
// the P*V operand is staged in transposed LDS from the very same key rows
// the scores consume, one global pass per 16-key tile).
// Waves: 4 per row block; a wave owns latent dims [128p, 128p+128) and rope
// dims [16p, 16p+16) for QK (9 hi/lo fragments) and its 128-dim slice of the
// 512-dim latent accumulator for P*V (8 fragments).
// ---------------------------------------------------------------------------
namespace w {

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kKeys = 16;
constexpr std::uint32_t kSlice = 128;         // latent dims per wave
constexpr std::uint32_t kSliceSteps = kSlice / 16;
constexpr std::uint32_t kWavesPerRow = 4;     // 512 / kSlice
constexpr std::uint32_t kLatStride = 520;     // K latent halves rows (pad 8)
constexpr std::uint32_t kRopeStride = 72;     // K rope halves rows (pad 8)
constexpr std::uint32_t kVtStride = 16;  // V^T rows, exact (4-way LDS)
constexpr std::uint32_t kPStride = 24;        // P rows (16B-aligned)

constexpr std::uint32_t cdiv(std::uint32_t a, std::uint32_t b) {
  return (a + b - 1U) / b;
}

template<std::uint32_t kHeads>
__global__ void __launch_bounds__(32U * kWavesPerRow * kHeads)
    WmmaMlaPrefill(const float* __restrict__ q_abs,
                   const float* __restrict__ q_rope,
                   const float* __restrict__ kv,
                   const float* __restrict__ kpe, float* __restrict__ latent,
                   std::uint32_t start_pos, std::uint32_t tokens,
                   std::uint32_t heads, std::uint32_t kv_rank,
                   std::uint32_t rotary, float scale) {
  constexpr std::uint32_t kRowBlocks = kHeads;
  constexpr std::uint32_t kThreads = 32U * kWavesPerRow * kHeads;
  __shared__ __align__(16) __half k_lat[kKeys][kLatStride];
  __shared__ __align__(16) __half k_lat_lo[kKeys][kLatStride];
  __shared__ __align__(16) __half k_rope[kKeys][kRopeStride];
  __shared__ __align__(16) __half k_rope_lo[kKeys][kRopeStride];
  __shared__ float s_lds[kRowBlocks][kWavesPerRow][16][16];
  __shared__ __align__(16) __half p_lds[kRowBlocks][2][16][kPStride];
  __shared__ float row_scale[kRowBlocks][16];
  __shared__ float row_sum[kRowBlocks][16];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 5U);
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t rb = wave / kWavesPerRow;
  const std::uint32_t part = wave % kWavesPerRow;
  const std::uint32_t dim0 = part * kSlice;
  const std::uint32_t rdim0 = part * 16U;

  const std::uint32_t q0 = blockIdx.x * 16U;
  const std::uint32_t head = blockIdx.y * kHeads + rb;
  const std::uint32_t q_last = min(q0 + 16U, tokens);
  const std::uint32_t causal_max = start_pos + q_last;

  uint4 q_hi_lat[kSliceSteps], q_lo_lat[kSliceSteps], q_hi_rope, q_lo_rope;

  const std::uint32_t row = q0 + sub;
  {
    const bool live = row < tokens;
    const std::size_t qo =
        (static_cast<std::size_t>(live ? row : q0) * heads + head) * kv_rank;
    const std::size_t ro =
        (static_cast<std::size_t>(live ? row : q0) * heads + head) * rotary;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
      const float* src = q_abs + qo + dim0 + ks * 16U + half_id * 8U;
      const float4 f0 = live ? *reinterpret_cast<const float4*>(src)
                             : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      const float4 f1 = live ? *reinterpret_cast<const float4*>(src + 4)
                             : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      const __half2 h[4] = {
          __floats2half2_rn(f0.x, f0.y), __floats2half2_rn(f0.z, f0.w),
          __floats2half2_rn(f1.x, f1.y), __floats2half2_rn(f1.z, f1.w)};
      __builtin_memcpy(&q_hi_lat[ks], h, 16);
      const float2 d0 = __half22float2(h[0]);
      const float2 d1 = __half22float2(h[1]);
      const float2 d2 = __half22float2(h[2]);
      const float2 d3 = __half22float2(h[3]);
      const __half2 l[4] = {__floats2half2_rn(f0.x - d0.x, f0.y - d0.y),
                            __floats2half2_rn(f0.z - d1.x, f0.w - d1.y),
                            __floats2half2_rn(f1.x - d2.x, f1.y - d2.y),
                            __floats2half2_rn(f1.z - d3.x, f1.w - d3.y)};
      __builtin_memcpy(&q_lo_lat[ks], l, 16);
    }
    {
      const float* src = q_rope + ro + rdim0 + half_id * 8U;
      const float4 f0 = live ? *reinterpret_cast<const float4*>(src)
                             : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      const float4 f1 = live ? *reinterpret_cast<const float4*>(src + 4)
                             : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
      const __half2 h[4] = {
          __floats2half2_rn(f0.x, f0.y), __floats2half2_rn(f0.z, f0.w),
          __floats2half2_rn(f1.x, f1.y), __floats2half2_rn(f1.z, f1.w)};
      __builtin_memcpy(&q_hi_rope, h, 16);
      const float2 d0 = __half22float2(h[0]);
      const float2 d1 = __half22float2(h[1]);
      const float2 d2 = __half22float2(h[2]);
      const float2 d3 = __half22float2(h[3]);
      const __half2 l[4] = {__floats2half2_rn(f0.x - d0.x, f0.y - d0.y),
                            __floats2half2_rn(f0.z - d1.x, f0.w - d1.y),
                            __floats2half2_rn(f1.x - d2.x, f1.y - d2.y),
                            __floats2half2_rn(f1.z - d3.x, f1.w - d3.y)};
      __builtin_memcpy(&q_lo_rope, l, 16);
    }
  }

  // Rebuild the duplicated-operand fragment from the owning 8 dims per lane
  // (gemma4 3.4a kGlobal trick, attention_wmma.hip.cpp:172).
  const auto q_frag = [&](const uint4& own) -> v16h {
    const auto swap = [](std::uint32_t v) {
      return static_cast<std::uint32_t>(__builtin_amdgcn_permlanex16(
          v, v, 0x76543210U, 0xFEDCBA98U, false, false));
    };
    const uint4 other =
        make_uint4(swap(own.x), swap(own.y), swap(own.z), swap(own.w));
    union {
      v16h f;
      uint4 u[2];
    } cvt;
    cvt.u[0] = half_id == 0 ? own : other;
    cvt.u[1] = half_id == 0 ? other : own;
    return cvt.f;
  };

  v8f o_acc[kSliceSteps] = {};

  // Online softmax: every wave runs its kWavesPerRow-th rows of the block.
  constexpr std::uint32_t kRowsPerPart = 16 / kWavesPerRow;
  constexpr std::uint32_t kSegLanes = 32 / kRowsPerPart;
  constexpr std::uint32_t kSegKeys = kKeys / kSegLanes;
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::uint32_t sm_row = part * kRowsPerPart + lane / kSegLanes;
  const std::uint32_t sm_seg = lane % kSegLanes;
  const std::uint32_t sm_query = q0 + sm_row;
  const std::uint32_t sm_keys = start_pos + min(sm_query, tokens - 1U) + 1U;

  // Staging (one global pass per tile): a thread owns dims [d4, d4+4) of two
  // consecutive keys, so the transposed V word packs locally with the same
  // f32x2 -> f16x2 conversion as the K hi/lo split; no cross-lane transpose.
  // Latent chunk c = tid + j*kThreads maps pair = c % 8, dg = c / 8: the v^T
  // word at [dim][pair] keeps banks split 4-deep across the warp, while each
  // 32-byte sector is consumed whole by a lane pair. Rope: one chunk per
  // lane, key = tid % 16, float4 slot = (tid / 16).
  constexpr std::uint32_t kLatStage = 4;  // latent chunks per thread-tile
  float4 a_lat[kLatStage], b_lat[kLatStage];  // keys 2p and 2p+1, dims d4
  float4 r_own;                               // rope float4, key tid % 16

  const auto fetch = [&](std::uint32_t key0) {
#pragma unroll
    for (std::uint32_t j = 0; j < kLatStage; ++j) {
      const std::uint32_t c = tid + j * kThreads;
      const std::uint32_t p = c % 8U;      // key pair
      const std::uint32_t dg = c / 8U;     // float4 group within the row
      const std::uint32_t base_key = key0 + 2U * p;
      const bool v0 = base_key < causal_max;
      const bool v1 = base_key + 1U < causal_max;
      const std::size_t o0 = static_cast<std::size_t>(v0 ? base_key : 0) *
                                 kv_rank +
                             dg * 4U;
      const std::size_t o1 = static_cast<std::size_t>(v1 ? base_key + 1U : 0) *
                                 kv_rank +
                             dg * 4U;
      a_lat[j] = v0 ? *reinterpret_cast<const float4*>(kv + o0)
                    : make_float4(0, 0, 0, 0);
      b_lat[j] = v1 ? *reinterpret_cast<const float4*>(kv + o1)
                    : make_float4(0, 0, 0, 0);
    }
    const std::uint32_t rk = tid % 16U;
    const std::uint32_t rd = (tid / 16U) * 4U;  // 16 float4 slots over 64 dims
    const bool v = key0 + rk < causal_max;
    r_own = v ? *reinterpret_cast<const float4*>(
                    kpe + static_cast<std::size_t>(key0 + rk) * rotary + rd)
              : make_float4(0, 0, 0, 0);
  };

  const auto h2u = [](__half2 v) {
    return *reinterpret_cast<const std::uint32_t*>(&v);
  };
  const auto store_k = [&] {
#pragma unroll
    for (std::uint32_t j = 0; j < kLatStage; ++j) {
      const std::uint32_t c = tid + j * kThreads;
      const std::uint32_t keyp = 2U * (c % 8U);
      const std::uint32_t d4 = (c / 8U) * 4U;
      const __half2 ha[2] = {__floats2half2_rn(a_lat[j].x, a_lat[j].y),
                             __floats2half2_rn(a_lat[j].z, a_lat[j].w)};
      const __half2 hb[2] = {__floats2half2_rn(b_lat[j].x, b_lat[j].y),
                             __floats2half2_rn(b_lat[j].z, b_lat[j].w)};
      const float2 da0 = __half22float2(ha[0]);
      const float2 da1 = __half22float2(ha[1]);
      const float2 db0 = __half22float2(hb[0]);
      const float2 db1 = __half22float2(hb[1]);
      const __half2 la[2] = {__floats2half2_rn(a_lat[j].x - da0.x,
                                               a_lat[j].y - da0.y),
                             __floats2half2_rn(a_lat[j].z - da1.x,
                                               a_lat[j].w - da1.y)};
      const __half2 lb[2] = {__floats2half2_rn(b_lat[j].x - db0.x,
                                               b_lat[j].y - db0.y),
                             __floats2half2_rn(b_lat[j].z - db1.x,
                                               b_lat[j].w - db1.y)};
      *reinterpret_cast<uint2*>(&k_lat[keyp][d4]) =
          *reinterpret_cast<const uint2*>(ha);
      *reinterpret_cast<uint2*>(&k_lat_lo[keyp][d4]) =
          *reinterpret_cast<const uint2*>(la);
      *reinterpret_cast<uint2*>(&k_lat[keyp + 1][d4]) =
          *reinterpret_cast<const uint2*>(hb);
      *reinterpret_cast<uint2*>(&k_lat_lo[keyp + 1][d4]) =
          *reinterpret_cast<const uint2*>(lb);
    }
    const std::uint32_t rk = tid % 16U;
    const std::uint32_t rd = (tid / 16U) * 4U;
    const __half2 h[2] = {__floats2half2_rn(r_own.x, r_own.y),
                          __floats2half2_rn(r_own.z, r_own.w)};
    const float2 d0 = __half22float2(h[0]);
    const float2 d1 = __half22float2(h[1]);
    const __half2 l[2] = {__floats2half2_rn(r_own.x - d0.x, r_own.y - d0.y),
                          __floats2half2_rn(r_own.z - d1.x, r_own.w - d1.y)};
    *reinterpret_cast<uint2*>(&k_rope[rk][rd]) =
        *reinterpret_cast<const uint2*>(h);
    *reinterpret_cast<uint2*>(&k_rope_lo[rk][rd]) =
        *reinterpret_cast<const uint2*>(l);
  };
  // Scores (bar 1) retire the K region, so it doubles as hi/lo V^T staging
  // for the CURRENT tile straight from the registers; PV then reads it below.
  const auto store_vt = [&] {
    auto* vw_hi = reinterpret_cast<std::uint32_t*>(&k_lat[0][0]);
    auto* vw_lo = reinterpret_cast<std::uint32_t*>(&k_lat_lo[0][0]);
#pragma unroll
    for (std::uint32_t j = 0; j < kLatStage; ++j) {
      const std::uint32_t c = tid + j * kThreads;
      const std::uint32_t p = c % 8U;
      const std::uint32_t d4 = (c / 8U) * 4U;
      const __half2 ha0 = __floats2half2_rn(a_lat[j].x, b_lat[j].x);
      const __half2 ha1 = __floats2half2_rn(a_lat[j].y, b_lat[j].y);
      const __half2 ha2 = __floats2half2_rn(a_lat[j].z, b_lat[j].z);
      const __half2 ha3 = __floats2half2_rn(a_lat[j].w, b_lat[j].w);
      const float2 fa0 = __half22float2(ha0);
      const float2 fa1 = __half22float2(ha1);
      const float2 fa2 = __half22float2(ha2);
      const float2 fa3 = __half22float2(ha3);
      vw_hi[(d4 + 0U) * (kVtStride / 2U) + p] = h2u(ha0);
      vw_hi[(d4 + 1U) * (kVtStride / 2U) + p] = h2u(ha1);
      vw_hi[(d4 + 2U) * (kVtStride / 2U) + p] = h2u(ha2);
      vw_hi[(d4 + 3U) * (kVtStride / 2U) + p] = h2u(ha3);
      vw_lo[(d4 + 0U) * (kVtStride / 2U) + p] =
          h2u(__floats2half2_rn(a_lat[j].x - fa0.x, b_lat[j].x - fa0.y));
      vw_lo[(d4 + 1U) * (kVtStride / 2U) + p] =
          h2u(__floats2half2_rn(a_lat[j].y - fa1.x, b_lat[j].y - fa1.y));
      vw_lo[(d4 + 2U) * (kVtStride / 2U) + p] =
          h2u(__floats2half2_rn(a_lat[j].z - fa2.x, b_lat[j].z - fa2.y));
      vw_lo[(d4 + 3U) * (kVtStride / 2U) + p] =
          h2u(__floats2half2_rn(a_lat[j].w - fa3.x, b_lat[j].w - fa3.y));
    }
  };

  fetch(0);
  store_k();
  __syncthreads();

  for (std::uint32_t key0 = 0; key0 < causal_max; key0 += kKeys) {
    const bool more = key0 + kKeys < causal_max;

    // Scores over this wave's 9 slices: three terms per 16-dim step, one
    // accumulator per operand class so the tiny hi*lo and lo*hi corrections
    // never round against the large hi*hi running sum, then a two-add tree.
    {
      v8f acc_hh = {};
      v8f acc_hl = {};
      v8f acc_lh = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kSliceSteps; ++ks) {
        const v16h bh =
            LoadFrag(&k_lat[sub][dim0 + ks * 16U]);
        const v16h bl =
            LoadFrag(&k_lat_lo[sub][dim0 + ks * 16U]);
        const v16h ah = q_frag(q_hi_lat[ks]);
        const v16h al = q_frag(q_lo_lat[ks]);
        acc_hh = Wmma(ah, bh, acc_hh);
        acc_hl = Wmma(ah, bl, acc_hl);
        acc_lh = Wmma(al, bh, acc_lh);
      }
      {
        const v16h bh = LoadFrag(&k_rope[sub][rdim0]);
        const v16h bl = LoadFrag(&k_rope_lo[sub][rdim0]);
        const v16h ah = q_frag(q_hi_rope);
        const v16h al = q_frag(q_lo_rope);
        acc_hh = Wmma(ah, bh, acc_hh);
        acc_hl = Wmma(ah, bl, acc_hl);
        acc_lh = Wmma(al, bh, acc_lh);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[rb][part][2 * i + half_id][sub] =
            (acc_hh[i] + acc_hl[i]) + acc_lh[i];
      }
    }
    __syncthreads();

    // Online softmax for this wave's rows; P goes to p_lds in half.
    {
      float vals[kSegKeys];
      float tile_max = -INFINITY;
#pragma unroll
      for (std::uint32_t m = 0; m < kSegKeys; ++m) {
        const std::uint32_t col = sm_seg * kSegKeys + m;
        const std::uint32_t key = key0 + col;
        float s = 0.0F;
#pragma unroll
        for (std::uint32_t p = 0; p < kWavesPerRow; ++p) {
          s += s_lds[rb][p][sm_row][col];
        }
        const bool valid = sm_query < q_last && key < sm_keys;
        vals[m] = valid ? s * scale : -INFINITY;
        tile_max = fmaxf(tile_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSegLanes; off <<= 1) {
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, static_cast<int>(off), 32));
      }
      const float next_max = fmaxf(running_max, tile_max);
      const float corr =
          next_max == -INFINITY ? 1.0F : expf(running_max - next_max);
      float tile_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kSegKeys; ++m) {
        const float e =
            vals[m] == -INFINITY ? 0.0F : expf(vals[m] - next_max);
        tile_sum += e;
        const __half eh = __float2half(e);
        const float ef = __half2float(eh);
        p_lds[rb][0][sm_row][sm_seg * kSegKeys + m] = eh;
        p_lds[rb][1][sm_row][sm_seg * kSegKeys + m] =
            __float2half(e - ef);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSegLanes; off <<= 1) {
        tile_sum +=
            __shfl_xor(tile_sum, static_cast<int>(off), 32);
      }
      running_max = next_max;
      running_sum = running_sum * corr + tile_sum;
      if (sm_seg == 0) {
        row_scale[rb][sm_row] = corr;
      }
    }
    // Scores (barrier above) retired the K region: V^T of the CURRENT tile
    // stages hi/lo from the registers into it, read after the next barrier.
    store_vt();
    if (more) {
      fetch(key0 + kKeys);
    }
    __syncthreads();

    float rsc[8];
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      rsc[i] = row_scale[rb][2 * i + half_id];
    }
    bool unchanged = true;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      unchanged = unchanged && (rsc[i] == 1.0F);
    }
    if (!__all(unchanged)) {
#pragma unroll
      for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[t][i] *= rsc[i];
        }
      }
    }
    const v16h p_hi = LoadFrag(&p_lds[rb][0][sub][0]);
    const v16h p_lo = LoadFrag(&p_lds[rb][1][sub][0]);
#pragma unroll 4
    for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
      const v16h vh =
          LoadFrag(reinterpret_cast<const __half*>(&k_lat[0][0]) +
                   (dim0 + t * 16U + sub) * kVtStride);
      const v16h vl =
          LoadFrag(reinterpret_cast<const __half*>(&k_lat_lo[0][0]) +
                   (dim0 + t * 16U + sub) * kVtStride);
      o_acc[t] = Wmma(p_hi, vh, o_acc[t]);
      o_acc[t] = Wmma(p_lo, vh, o_acc[t]);
      o_acc[t] = Wmma(p_hi, vl, o_acc[t]);
    }
    __syncthreads();
    // PV retired this tile's V; the registers retired at store_vt above.
    // Prefetch, K-stage and barrier complete the 4-sync cadence.
    if (more) {
      store_k();
      __syncthreads();
    }
  }

  if (sm_seg == 0) {
    row_sum[rb][sm_row] = running_sum;
  }
  __syncthreads();
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t r = 2 * i + half_id;
    const std::uint32_t token = q0 + r;
    if (token >= tokens) {
      continue;
    }
    const float l = row_sum[rb][r];
    const float inv = l > 0.0F ? 1.0F / l : 0.0F;
    const std::size_t at =
        (static_cast<std::size_t>(token) * heads + head) * kv_rank + dim0 +
        sub;
#pragma unroll
    for (std::uint32_t t = 0; t < kSliceSteps; ++t) {
      latent[at + t * 16U] = o_acc[t][i] * inv;
    }
  }
}

}  // namespace w

// Runs candidate 0 = shipped fused, 1 = FusedKVReuse.
void RunCand(int cand, const Buffers& b, std::uint32_t start_pos,
             std::uint32_t tokens) {
  const float scale = 1.0F / std::sqrt(192.0F);
  const std::uint32_t q_tiles =
      (tokens + q::kPrefillFusedQTile - 1U) / q::kPrefillFusedQTile;
  if (cand == 0) {
    q::MlaPrefillFusedKernel<q::kPrefillFusedKvSlot>
        <<<dim3(kHeads, q_tiles), q::kPrefillFusedWarps * 32U>>>(
            b.q, b.qr, b.kv, b.kpe, b.out, start_pos, tokens, kHeads, kKvRank,
            kRot, scale);
  } else if (cand == 1) {
    FusedKVReuse<q::kPrefillFusedKvSlot, 8U, 4U>
        <<<dim3(kHeads, q_tiles), 8U * 32U>>>(
            b.q, b.qr, b.kv, b.kpe, b.out, start_pos, tokens, kHeads, kKvRank,
            kRot, scale);
  } else if (cand == 2) {
    const std::uint32_t qt = (tokens + 32U - 1U) / 32U;
    (void)q_tiles;
    FusedKVReuse<q::kPrefillFusedKvSlot, 16U, 2U>
        <<<dim3(kHeads, qt), 16U * 32U>>>(
            b.q, b.qr, b.kv, b.kpe, b.out, start_pos, tokens, kHeads, kKvRank,
            kRot, scale);
  } else if (cand == 4) {
    w::WmmaMlaPrefill<2U><<<dim3((tokens + 15U) / 16U, kHeads / 2U), 256U>>>(
        b.q, b.qr, b.kv, b.kpe, b.out, start_pos, tokens, kHeads, kKvRank,
        kRot, scale);
  } else {
    const std::uint32_t qt = (tokens + 64U - 1U) / 64U;
    (void)q_tiles;
    FusedKVReuse<q::kPrefillFusedKvSlot, 16U, 4U>
        <<<dim3(kHeads, qt), 16U * 32U>>>(
            b.q, b.qr, b.kv, b.kpe, b.out, start_pos, tokens, kHeads, kKvRank,
            kRot, scale);
  }
}

double TimeCand(int cand, const Buffers& b, std::uint32_t start_pos,
                std::uint32_t tokens) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  RunCand(cand, b, start_pos, tokens);
  CheckCuda(hipDeviceSynchronize(), "warm");
  float best = 1e30F;
  for (int r = 0; r < 3; ++r) {
    hipEventRecord(e0);
    for (int i = 0; i < 5; ++i) {
      RunCand(cand, b, start_pos, tokens);
    }
    hipEventRecord(e1);
    CheckCuda(hipDeviceSynchronize(), "sync");
    float ms = 0.0F;
    hipEventElapsedTime(&ms, e0, e1);
    best = ms < best ? ms : best;
  }
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  return static_cast<double>(best) * 1000.0 / 5.0;
}

std::vector<float> Random(std::size_t n, std::uint32_t seed);

int Cand() {
  // Shapes from real prefill: 4096-ctx chunks of 1024 and a 16384-ctx
  // sample. Correctness gate: bit-identical to the shipped fused kernel.
  for (auto sh : {std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>{
                      1024U, 4096U, 16U},
                  {2048U, 16384U, 16U}, {2048U, 16384U, 100U}}) {
    const std::uint32_t tok = std::get<0>(sh);
    const std::uint32_t ctx = std::get<1>(sh);
    const std::uint32_t nchunk = ctx / tok;
    auto q_abs = Random(static_cast<std::size_t>(tok) * kHeads * kKvRank, 7);
    auto q_rope = Random(static_cast<std::size_t>(tok) * kHeads * kRot, 8);
    auto kv = Random(static_cast<std::size_t>(ctx) * kKvRank, 9);
    auto kpe = Random(static_cast<std::size_t>(ctx) * kRot, 10);
    for (std::uint32_t c = 0; c < nchunk; ++c) {
      const std::uint32_t start = c * tok;
      if (c + 1U < nchunk && c % 4U != 3U) {
        continue;  // sample chunk depths
      }
      Buffers b = MakeBuffers(ctx, tok, q_abs, q_rope, kv, kpe);
      RunCand(0, b, start, tok);
      CheckCuda(hipDeviceSynchronize(), "r0");
      std::vector<float> got0(static_cast<std::size_t>(tok) * kHeads *
                                  kKvRank,
                              0.0F);
      CheckCuda(hipMemcpy(got0.data(), b.out, got0.size() * 4,
                          hipMemcpyDeviceToHost),
                "o0");
      const double t0 = TimeCand(0, b, start, tok);
      for (int v = 1; v <= 4; ++v) {
        (void)hipMemset(b.out, 0, got0.size() * 4);
        RunCand(v, b, start, tok);
        CheckCuda(hipDeviceSynchronize(), "r1");
        std::vector<float> got1(got0.size(), 0.0F);
        CheckCuda(hipMemcpy(got1.data(), b.out, got1.size() * 4,
                            hipMemcpyDeviceToHost),
                  "o1");
        double bits = 0.0;
        double nans = 0.0;
        double max_rel = 0.0;
        for (std::size_t i = 0; i < got0.size(); ++i) {
          if (std::isnan(got0[i]) || std::isnan(got1[i])) {
            nans += 1.0;
          }
          if (got0[i] != got1[i]) {
            bits += 1.0;
            const double m = std::fabs(got1[i] - got0[i]);
            max_rel = m > max_rel ? m : max_rel;
          }
        }
        const double tv = TimeCand(v, b, start, tok);
        std::printf(
            "  tok=%4u ctx=%5u start=%5u v%d  bitdiff=%.0f/%zu maxabs %.1e"
            " nans %.0f   fused %9.1f -> %9.1f us  gain %5.2fx\n",
            tok, ctx, start, v, bits, got0.size(), max_rel, nans, t0, tv,
            t0 / tv);
      }
      hipFree(b.q);
      hipFree(b.qr);
      hipFree(b.kv);
      hipFree(b.kpe);
      hipFree(b.out);
    }
  }
  return 0;
}

// Replay the real engine-dumped (GUFO_MLA_DUMP) chunk records: one line in
// meta per record followed by q, q_rope, kv, kpe float blocks in data file.
int ReplayReal(const std::string& data_path, const std::string& meta_path) {
  std::FILE* fd = std::fopen(data_path.c_str(), "rb");
  std::FILE* fm = std::fopen(meta_path.c_str(), "r");
  if (fd == nullptr || fm == nullptr) {
    std::printf("replay real: cannot open %s\n", data_path.c_str());
    return 1;
  }
  std::printf("replay real: %s\n", data_path.c_str());
  char line[256];
  for (bool fused_var : {true, false}) {
    std::rewind(fm);
    double tsum = 0.0;
    while (std::fgets(line, sizeof(line), fm) != nullptr) {
      std::uint32_t start = 0, tok = 0, heads = 0, kvm = 0, rot = 0;
      float scale = 0.0F;
      if (std::sscanf(line, "%u %u %u %u %u %g", &start, &tok, &heads, &kvm,
                      &rot, &scale) != 6) {
        continue;
      }
      const std::size_t rows = static_cast<std::size_t>(start) + tok;
      std::vector<float> q(static_cast<std::size_t>(tok) * heads * kvm);
      std::vector<float> qr(static_cast<std::size_t>(tok) * heads * rot);
      std::vector<float> kvv(rows * kvm);
      std::vector<float> kpe(rows * rot);
      if (std::fread(q.data(), sizeof(float), q.size(), fd) != q.size() ||
          std::fread(qr.data(), sizeof(float), qr.size(), fd) != qr.size() ||
          std::fread(kvv.data(), sizeof(float), kvv.size(), fd) != kvv.size() ||
          std::fread(kpe.data(), sizeof(float), kpe.size(), fd) != kpe.size()) {
        std::printf("  short read at start=%u\n", start);
        return 1;
      }
      Buffers b = MakeBuffers(rows, tok, q, qr, kvv, kpe);
      RunVariant(fused_var, b, start, tok);
      CheckCuda(hipDeviceSynchronize(), "warm");
      float best = 1e30F;
      for (int r = 0; r < 3; ++r) {
        hipEvent_t e0, e1;
        hipEventCreate(&e0);
        hipEventCreate(&e1);
        hipEventRecord(e0);
        RunVariant(fused_var, b, start, tok);
        hipEventRecord(e1);
        CheckCuda(hipDeviceSynchronize(), "run");
        float ms = 0.0F;
        hipEventElapsedTime(&ms, e0, e1);
        best = ms < best ? ms : best;
        hipEventDestroy(e0);
        hipEventDestroy(e1);
      }
      tsum += best;
      std::printf("  %-5s start=%5u tok=%u %.1f ms\n",
                  fused_var ? "fused" : "old", start, tok, best);
      hipFree(b.q);
      hipFree(b.qr);
      hipFree(b.kv);
      hipFree(b.kpe);
      hipFree(b.out);
    }
    std::printf("total %s %.0f ms\n", fused_var ? "fused" : "old", tsum);
  }
  std::fclose(fd);
  std::fclose(fm);
  return 0;
}

void RunVariant(bool fused, const Buffers& b, std::uint32_t start_pos,
                std::uint32_t tokens) {
  const float scale = 1.0F / std::sqrt(192.0F);
  if (fused) {
    const std::uint32_t q_tiles =
        (tokens + q::kPrefillFusedQTile - 1U) / q::kPrefillFusedQTile;
    q::MlaPrefillFusedKernel<q::kPrefillFusedKvSlot>
        <<<dim3(kHeads, q_tiles), q::kPrefillFusedWarps * 32U>>>(
            b.q, b.qr, b.kv, b.kpe, b.out, start_pos, tokens, kHeads, kKvRank,
            kRot, scale);
  } else {
    const std::uint32_t q_tiles =
        (tokens + q::kPrefillQT - 1U) / q::kPrefillQT;
    q::MlaPrefillKernel<<<dim3(kHeads, q_tiles),
                          kKvRank > q::kPrefillQT * q::kPrefillKT
                              ? kKvRank
                              : q::kPrefillQT * q::kPrefillKT>>>(
        b.q, b.qr, b.kv, b.kpe, b.out, start_pos, tokens, kHeads, kKvRank,
        kRot, scale);
  }
}

double TimeRun(bool fused, const Buffers& b, std::uint32_t start_pos,
               std::uint32_t tokens, int reps) {
  hipEvent_t e0, e1;
  CheckCuda(hipEventCreate(&e0), "event");
  CheckCuda(hipEventCreate(&e1), "event");
  RunVariant(fused, b, start_pos, tokens);
  CheckCuda(hipDeviceSynchronize(), "warm");
  CheckCuda(hipEventRecord(e0), "rec0");
  for (int i = 0; i < reps; ++i) {
    RunVariant(fused, b, start_pos, tokens);
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

// Replay the per-call shape sequence of a pp4096 prefill pass (4 causal
// chunks, back-to-back) with optional L2 flushes to reproduce e2e
// cache-residency conditions and print every per-call time.
int Replay(const std::string& args) {
  const bool flush = args.find('f') != std::string::npos;
  const bool big = args.find('b') != std::string::npos;
  const bool huge_alloc = args.find('a') != std::string::npos;
  const std::uint32_t tokens = 1024U;
  const std::uint32_t rows = 4096U;
  float* scratch = nullptr;
  if (huge_alloc) {
    // Emulate a resident 70 GiB model so the page tables are under pressure.
    const std::size_t elems = static_cast<std::size_t>(70) * 1024U * 1024U *
                              1024U / 4U;
    scratch = Dev(elems);
    CheckCuda(hipMemset(scratch, 1, elems * 4U), "scratch");
    if (std::getenv("GUFO_BENCH_TOUCH") != nullptr) {
      CheckCuda(hipMemset(scratch, 2, elems * 4U), "scratch2");
    }
  }
  auto q_abs = Random(static_cast<std::size_t>(tokens) * kHeads * kKvRank, 7);
  auto q_rope = Random(static_cast<std::size_t>(tokens) * kHeads * kRot, 8);
  auto kv = Random(static_cast<std::size_t>(rows) * kKvRank, 9);
  auto kpe = Random(static_cast<std::size_t>(rows) * kRot, 10);
  if (big) {
    // Activations scaled so QK scores reach +-1e4, like real MLA states.
    for (float& x : q_abs) x *= 24.0F;
    for (float& x : q_rope) x *= 24.0F;
    for (float& x : kv) x *= 24.0F;
    for (float& x : kpe) x *= 24.0F;
  }
  float* cadence_buf = nullptr;
  std::size_t cadence_elems = 0U;
  if (const char* cadence = std::getenv("GUFO_BENCH_CADENCE")) {
    cadence_elems = static_cast<std::size_t>(std::atoi(cadence)) * 1024U *
                    1024U / 4U;
    cadence_buf = Dev(cadence_elems);
  }
  if (const char* real = std::getenv("GUFO_BENCH_REAL")) {
    return ReplayReal(real, std::getenv("GUFO_BENCH_REAL_META") ?
                           std::getenv("GUFO_BENCH_REAL_META") : real);
  }
  Buffers b = MakeBuffers(rows, tokens, q_abs, q_rope, kv, kpe);
  float* flush_buf = flush ? Dev(64U * 1024U * 1024U / 4U) : nullptr;
  std::printf("replay flush=%d big=%d (%s):\n", flush ? 1 : 0, big ? 1 : 0,
              flush ? "L2-flushed" : "L2-warm");
  for (bool fused_var : {false, true}) {
    RunVariant(fused_var, b, 0U, tokens);
    CheckCuda(hipDeviceSynchronize(), "warm");
    for (std::uint32_t pass = 0; pass < 2U; ++pass) {
      for (std::uint32_t layer = 0; layer < 2U; ++layer) {
        for (std::uint32_t chunk = 0; chunk < 4U; ++chunk) {
          if (flush) {
            CheckCuda(hipMemsetAsync(flush_buf, chunk, 64U * 1024U * 1024U),
                      "flush");
          }
          if (cadence_buf != nullptr) {
            CheckCuda(hipMemsetAsync(cadence_buf, 3, cadence_elems),
                      "cadence1");
            CheckCuda(hipMemsetAsync(cadence_buf, 5, cadence_elems),
                      "cadence2");
          }
          hipEvent_t e0, e1;
          CheckCuda(hipEventCreate(&e0), "e0");
          CheckCuda(hipEventCreate(&e1), "e1");
          CheckCuda(hipEventRecord(e0), "r0");
          RunVariant(fused_var, b, chunk * tokens, tokens);
          CheckCuda(hipEventRecord(e1), "r1");
          CheckCuda(hipEventSynchronize(e1), "sy");
          float ms = 0.0F;
          CheckCuda(hipEventElapsedTime(&ms, e0, e1), "el");
          std::printf("  %s %5.1f ms\n", fused_var ? "fused" : "old  ",
                      static_cast<double>(ms));
          hipEventDestroy(e0);
          hipEventDestroy(e1);
        }
        (void)layer;
      }
    }
  }
  if (flush) {
    hipFree(flush_buf);
  }
  hipFree(b.q);
  hipFree(b.qr);
  hipFree(b.kv);
  hipFree(b.kpe);
  hipFree(b.out);
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "replay") {
    return Replay(argc > 2 ? std::string(argv[2]) : std::string());
  }
  if (argc > 1 && std::string(argv[1]) == "cand") {
    return Cand();
  }
  if (argc > 1 && std::string(argv[1]) == "wmma") {
    // fp64-oracle gate for candidate 4 on small shapes (timing meaningless
    // here; kernel gate compares against the shipped fused output).
    bool ok = true;
    for (auto st : {std::pair<std::uint32_t, std::uint32_t>{0U, 1U},
                    {0U, 7U}, {5U, 32U}, {130U, 64U}, {400U, 33U},
                    {0U, 100U}, {17U, 512U}, {0U, 512U}, {64U, 16U}}) {
      const std::uint32_t start = st.first;
      const std::uint32_t tokens = st.second;
      const std::uint32_t rows = start + tokens;
      auto q_abs = Random(static_cast<std::size_t>(tokens) * kHeads * kKvRank,
                          11 + tokens + 7 * start);
      auto q_rope = Random(static_cast<std::size_t>(tokens) * kHeads * kRot,
                           21 + tokens + 7 * start);
      auto kv = Random(static_cast<std::size_t>(rows) * kKvRank, 31 + rows);
      auto kpe = Random(static_cast<std::size_t>(rows) * kRot, 41 + rows);
      std::vector<float> ref;
      Oracle(q_abs, q_rope, kv, kpe, start, tokens,
             1.0F / std::sqrt(192.0F), &ref);
      Buffers b = MakeBuffers(rows, tokens, q_abs, q_rope, kv, kpe);
      RunCand(4, b, start, tokens);
      CheckCuda(hipDeviceSynchronize(), "run");
      std::vector<float> got(ref.size());
      CheckCuda(hipMemcpy(got.data(), b.out, got.size() * 4,
                          hipMemcpyDeviceToHost),
                "out");
      // fp16-P quantization puts absolute error at ~1e-4 of the row scale;
      // gate relative to each (token, head) row's magnitude, not raw fp64.
      float worst = 0.0F;
      std::size_t wi = 0;
      std::size_t nan = 0;
      for (std::size_t e = 0; e < got.size(); ++e) {
        if (std::isnan(got[e])) {
          ++nan;
        }
      }
      for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t h = 0; h < kHeads; ++h) {
          const std::size_t base =
              (static_cast<std::size_t>(t) * kHeads + h) * kKvRank;
          float rmax = 0.0F;
          for (std::uint32_t r = 0; r < kKvRank; ++r) {
            rmax = fmaxf(rmax, std::fabs(ref[base + r]));
          }
          for (std::uint32_t r = 0; r < kKvRank; ++r) {
            const float rel =
                std::fabs(got[base + r] - ref[base + r]) / (rmax + 1e-6F);
            if (rel > worst) {
              worst = rel;
              wi = base + r;
            }
          }
        }
      }
      const ErrStat stat = WorstRelative(ref, got);
      const double wabs = std::fabs(got[stat.worst_idx] - ref[stat.worst_idx]);
      RunCand(0, b, start, tokens);
      CheckCuda(hipDeviceSynchronize(), "run fused");
      std::fill(got.begin(), got.end(), 0.0F);
      CheckCuda(hipMemcpy(got.data(), b.out, got.size() * 4,
                          hipMemcpyDeviceToHost),
                "out fused");
      const ErrStat fstat = WorstRelative(ref, got);
      const double fabsd = std::fabs(got[fstat.worst_idx] - ref[fstat.worst_idx]);
      const bool pass = worst <= 2e-3 && nan == 0U;
      ok = ok && pass;
      std::printf("  start=%5u tok=%4u wmma rowrel %.3e ctest %.3e"
                  " (abs %.2e @ref %+.4f) fused ctest %.3e (abs %.2e"
                  " @ref %+.4f) nan %zu %s  [t=%u h=%u r=%u ref %+.6f"
                  " got %+.6f]\n",
                  start, tokens, worst, stat.worst, wabs,
                  static_cast<double>(ref[stat.worst_idx]), fstat.worst, fabsd,
                  static_cast<double>(ref[fstat.worst_idx]),
                  nan, pass ? "OK" : "FAIL",
                  static_cast<std::uint32_t>(wi / (kHeads * kKvRank)),
                  static_cast<std::uint32_t>((wi / kKvRank) % kHeads),
                  static_cast<std::uint32_t>(wi % kKvRank), ref[wi], got[wi]);
      hipFree(b.q);
      hipFree(b.qr);
      hipFree(b.kv);
      hipFree(b.kpe);
      hipFree(b.out);
    }
    return ok ? 0 : 1;
  }
  const bool quick = argc > 1;
  bool ok = true;
  std::printf("correctness:\n");
  for (auto st : {std::pair<std::uint32_t, std::uint32_t>{0U, 1U},
                  {0U, 7U}, {5U, 32U}, {130U, 64U}, {400U, 33U}, {0U, 100U},
                  {17U, 512U}}) {
    const std::uint32_t start = st.first;
    const std::uint32_t tokens = st.second;
    const std::uint32_t rows = start + tokens;
    auto q_abs = Random(static_cast<std::size_t>(tokens) * kHeads * kKvRank,
                        11 + tokens + 7 * start);
    auto q_rope = Random(static_cast<std::size_t>(tokens) * kHeads * kRot,
                         21 + tokens + 7 * start);
    auto kv = Random(static_cast<std::size_t>(rows) * kKvRank, 31 + rows);
    auto kpe = Random(static_cast<std::size_t>(rows) * kRot, 41 + rows);
    std::vector<float> ref;
    Oracle(q_abs, q_rope, kv, kpe, start, tokens, 1.0F / std::sqrt(192.0F),
           &ref);
    for (bool fused : {false, true}) {
      Buffers b = MakeBuffers(rows, tokens, q_abs, q_rope, kv, kpe);
      RunVariant(fused, b, start, tokens);
      CheckCuda(hipDeviceSynchronize(), "run");
      std::vector<float> got(ref.size());
      CheckCuda(hipMemcpy(got.data(), b.out, got.size() * 4,
                          hipMemcpyDeviceToHost),
                "out");
      const ErrStat stat = WorstRelative(ref, got);
      const double err = stat.worst;
      const bool pass = err <= 1e-2 && stat.nan == 0U;
      const std::size_t wi = stat.worst_idx;
      std::printf(
          "  start=%5u tok=%4u %-5s worst %.3e nan %zu %s"
          "  [i=%zu t=%u h=%u r=%u ref %+.6f got %+.6f]\n",
          start, tokens, fused ? "fused" : "old", err, stat.nan,
          pass ? "OK" : "FAIL", wi,
          static_cast<std::uint32_t>(wi / (kHeads * kKvRank)),
          static_cast<std::uint32_t>((wi / kKvRank) % kHeads),
          static_cast<std::uint32_t>(wi % kKvRank), ref[wi], got[wi]);
      hipFree(b.q);
      hipFree(b.qr);
      hipFree(b.kv);
      hipFree(b.kpe);
      hipFree(b.out);
    }
  }
  if (quick) {
    return ok ? 0 : 1;
  }
  std::printf("timing (us per MLA prefill call, all 32 heads):\n");
  for (auto st : {std::pair<std::uint32_t, std::uint32_t>{0U, 2048U},
                  {0U, 8192U}, {0U, 2048U}, {8192U, 1024U},
                  {15360U, 1024U}}) {
    const std::uint32_t start = st.first;
    const std::uint32_t tokens = st.second;
    const std::uint32_t rows = start + tokens;
    auto q_abs = Random(static_cast<std::size_t>(tokens) * kHeads * kKvRank,
                        77 + tokens + 7 * start);
    auto q_rope = Random(static_cast<std::size_t>(tokens) * kHeads * kRot,
                         88 + tokens + 7 * start);
    auto kv = Random(static_cast<std::size_t>(rows) * kKvRank, 99 + rows);
    auto kpe = Random(static_cast<std::size_t>(rows) * kRot, 61 + rows);
    Buffers b = MakeBuffers(rows, tokens, q_abs, q_rope, kv, kpe);
    const double old_us = TimeRun(false, b, start, tokens, 5);
    const double new_us = TimeRun(true, b, start, tokens, 5);
    const double flop = 2.0 * static_cast<double>(tokens) * kHeads *
                        (static_cast<double>(rows) + start + 1.0) / 2.0 *
                        (kKvRank + kRot + kKvRank);
    std::printf(
        "  start=%5u tok=%5u old %10.1f  fused %9.1f  speedup %6.1fx  "
        "fused TFLOP/s(min) %.2f\n",
        start, tokens, old_us, new_us, old_us / new_us,
        flop / (new_us * 1e6));
    hipFree(b.q);
    hipFree(b.qr);
    hipFree(b.kv);
    hipFree(b.kpe);
    hipFree(b.out);
  }
  return ok ? 0 : 1;
}
