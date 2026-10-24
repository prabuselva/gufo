// Nemotron Bolt K-quant decode GEMV bench (gfx1151). Decodes run
// GemvKqPackedKernel with warp-owned super-blocks (lane owns 8 consecutive
// values, LoadBytes8 funnel-shifts the Q6_K 210 B stride). Measured decode
// traffic is ~98 GB/s vs the ~240 GB/s sustained-copy ceiling, so probe:
//
//   v0  production replica (baseline, exact fold).
//   vread  pure weight-read sum at the exact production addresses (ceiling).
//   v1  two independent super-block chains per loop iteration (latency).
//   v2  re-aligned loads: a lane window is one contiguous 8-byte run; the
//       Q4_K/Q5_K/IQ4_XS strides (144/176/136) and every in-block offset are
//       multiples of 8, so a single aligned u64 replaces LoadBytes8. For
//       Q6_K (stride 210) an aligned u64 PAIR + 64-bit funnel re-aligns; the
//       pair stays inside the 210 B block, unlike the dword+slack window.
//   v3  v2 + factored ds (one mul per super-block instead of per value) and
//       the magic-number int->float conversion (2 full-rate VALU ops instead
//       of a quarter-rate v_cvt).
//   v4  v1 + v3.
//
// Reordering and factoring change rounding at ULP scale (the production
// contract is rel < 0.02 + argmax); the bench checks every variant against
// v0 to 1e-3 on random data.
//
//   tools/bench/build.sh nem_decode_kq_bench && /tmp/nem_decode_kq_bench

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cmd)                                                     \
  do {                                                                 \
    hipError_t err__ = (cmd);                                          \
    if (err__ != hipSuccess) {                                         \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err__), \
              __FILE__, __LINE__);                                     \
      exit(1);                                                         \
    }                                                                  \
  } while (0)

namespace {

constexpr int kBlock = 256;
constexpr int kWarp = 32;
constexpr int kKqElems = 256;

// GGML block types mirrored from nb_ops.hip.cpp.
struct NblQ4_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};
struct NblQ5_K {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qh[32];
  uint8_t qs[128];
};
struct NblQ6_K {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  uint16_t d;
};

// Tier ids mirroring kNblGemvQ4_K/Q5_K/Q6_K.
enum Tier { kQ4 = 5, kQ5 = 6, kQ6 = 7 };

__device__ __forceinline__ float Fp16ToF32(uint16_t h) {
  __half_raw r;
  r.x = h;
  return __half2float(*reinterpret_cast<const __half*>(&r));
}

__device__ __forceinline__ void QScaleMin(int i, const uint8_t* sc, float d,
                                          float dmin, float& s, float& m) {
  const int a = i / 2;
  const int b = i % 2;
  const int s0 = (sc[a] >> (4 * b)) & 0xF;
  s = d * static_cast<float>(s0);
  m = dmin * static_cast<float>(s0);
}

__device__ __forceinline__ uint64_t LoadU64(const uint8_t* w8) {
  const uint64_t u = *reinterpret_cast<const uint64_t*>(w8);
  return u;
}

// Re-align 8 bytes from the enclosing 16-byte window with a 64-bit funnel.
// The byte residue is warp-uniform (lane offsets are multiples of 8).
__device__ __forceinline__ uint64_t LoadBytes8U64(const uint8_t* w8) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(w8);
  const uint64_t* const wd =
      reinterpret_cast<const uint64_t*>(a & ~uintptr_t(7));
  const uint32_t sh = static_cast<uint32_t>(a & 7U) * 8U;
  const uint64_t d0 = wd[0];
  const uint64_t d1 = wd[1];
  if (sh == 0U) {
    return d0;
  }
  return (d0 >> sh) | (d1 << (64 - sh));
}

__device__ __forceinline__ float U32ToF32Magic(uint32_t u) {
  // u < 64: bitcast(u | 2^23) - 2^23, two full-rate VALU ops.
  return __uint_as_float(u | 0x4B000000U) - 8388608.0F;
}

// ---------------------------------------------------------------- variant 0

__device__ __forceinline__ uint2 LoadBytes8(const uint8_t* w8) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(w8);
  const uint32_t* const wd =
      reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
  const uint32_t sh = static_cast<uint32_t>(a & 3U) * 8U;
  const uint32_t d0 = wd[0];
  const uint32_t d1 = wd[1];
  if (sh == 0U) {
    return make_uint2(d0, d1);
  }
  const uint32_t d2 = wd[2];
  return make_uint2(__funnelshift_r(d0, d1, sh), __funnelshift_r(d1, d2, sh));
}

template<int TYPE>
__device__ __forceinline__ float WarpDotV0(const uint8_t* w, const float* x,
                                           int block, int lane) {
  if (TYPE == kQ6) {
    const NblQ6_K* b = reinterpret_cast<const NblQ6_K*>(w) + block;
    const int c = lane >> 4;
    const int k = (lane >> 2) & 3;
    const int hi16 = (lane >> 1) & 1;
    const int j0 = (lane & 3) * 8;
    const uint2 h8 = LoadBytes8(b->qh + c * 32 + j0);
    const uint2 q8 = LoadBytes8(b->ql + c * 64 + (k & 1) * 32 + j0);
    const float ds =
        Fp16ToF32(b->d) * static_cast<float>(b->scales[c * 8 + 2 * k + hi16]);
    const int sh_h = 2 * k;
    const int sh_q = (k >> 1) * 4;
    const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
    const float4 xa = *reinterpret_cast<const float4*>(xv);
    const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
    const uint32_t hw[2] = {h8.x, h8.y};
    const uint32_t qw[2] = {q8.x, q8.y};
    float acc = 0.0F;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint32_t h56 = hw[p >> 2] >> (8 * (p & 3));
      const uint32_t lo4 = qw[p >> 2] >> (8 * (p & 3));
      const int t = static_cast<int>(((h56 >> sh_h) & 0x3U) << 4U) +
                    static_cast<int>((lo4 >> sh_q) & 0x0FU) - 32;
      const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
      acc += ds * static_cast<float>(t) * xv8;
    }
    return acc;
  }
  if (TYPE == kQ5) {
    const NblQ5_K* b = reinterpret_cast<const NblQ5_K*>(w) + block;
    const float d = Fp16ToF32(b->d);
    const float dmin = Fp16ToF32(b->dmin);
    const int v0 = lane * 8;
    const int gg = v0 >> 6;
    const int half = (v0 >> 5) & 1;
    const int l0 = v0 & 31;
    float s = 0.0F;
    float m = 0.0F;
    QScaleMin(2 * gg + half, b->scales, d, dmin, s, m);
    const uint2 q8 = LoadBytes8(b->qs + gg * 32 + l0);
    const uint2 h8 = LoadBytes8(b->qh + l0);
    const int sh = 2 * gg + half;
    const uint32_t qw[2] = {q8.x, q8.y};
    const uint32_t hw[2] = {h8.x, h8.y};
    const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
    const float4 xa = *reinterpret_cast<const float4*>(xv);
    const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
    float acc = 0.0F;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const int b8 = 8 * (p & 3);
      const uint32_t q = (qw[p >> 2] >> b8) & 0xFFU;
      const uint32_t hi = (hw[p >> 2] >> (b8 + sh)) & 1U;
      const int v = static_cast<int>(half ? (q >> 4) : (q & 0x0FU)) +
                    static_cast<int>(hi << 4U);
      const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
      acc += (s * static_cast<float>(v) - m) * xv8;
    }
    return acc;
  }
  const NblQ4_K* b = reinterpret_cast<const NblQ4_K*>(w) + block;
  const float d = Fp16ToF32(b->d);
  const float dmin = Fp16ToF32(b->dmin);
  const int g = lane >> 2;
  float s = 0.0F;
  float m = 0.0F;
  QScaleMin(g, b->scales, d, dmin, s, m);
  const uint2 q8 = LoadBytes8(b->qs + (g >> 1) * 32 + (lane & 3) * 8);
  const int sh = (g & 1) * 4;
  const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
  const float4 xa = *reinterpret_cast<const float4*>(xv);
  const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
  const uint32_t qw[2] = {q8.x, q8.y};
  float acc = 0.0F;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint32_t byte = qw[p >> 2] >> (8 * (p & 3));
    const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
    acc += (s * static_cast<float>((byte >> sh) & 0x0FU) - m) * xv8;
  }
  return acc;
}

// ---------------------------------------------------------------- variant 2

template<int TYPE>
__device__ __forceinline__ float WarpDotV2(const uint8_t* w, const float* x,
                                           int block, int lane) {
  if (TYPE == kQ6) {
    const NblQ6_K* b = reinterpret_cast<const NblQ6_K*>(w) + block;
    const int c = lane >> 4;
    const int k = (lane >> 2) & 3;
    const int hi16 = (lane >> 1) & 1;
    const int j0 = (lane & 3) * 8;
    const uint64_t h8 = LoadBytes8U64(b->qh + c * 32 + j0);
    const uint64_t q8 = LoadBytes8U64(b->ql + c * 64 + (k & 1) * 32 + j0);
    const float ds =
        Fp16ToF32(b->d) * static_cast<float>(b->scales[c * 8 + 2 * k + hi16]);
    const int sh_h = 2 * k;
    const int sh_q = (k >> 1) * 4;
    const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
    const float4 xa = *reinterpret_cast<const float4*>(xv);
    const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
    const uint32_t hw[2] = {static_cast<uint32_t>(h8),
                            static_cast<uint32_t>(h8 >> 32)};
    const uint32_t qw[2] = {static_cast<uint32_t>(q8),
                            static_cast<uint32_t>(q8 >> 32)};
    float acc = 0.0F;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint32_t h56 = hw[p >> 2] >> (8 * (p & 3));
      const uint32_t lo4 = qw[p >> 2] >> (8 * (p & 3));
      const int t = static_cast<int>(((h56 >> sh_h) & 0x3U) << 4U) +
                    static_cast<int>((lo4 >> sh_q) & 0x0FU) - 32;
      const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
      acc += ds * static_cast<float>(t) * xv8;
    }
    return acc;
  }
  if (TYPE == kQ5) {
    const NblQ5_K* b = reinterpret_cast<const NblQ5_K*>(w) + block;
    const float d = Fp16ToF32(b->d);
    const float dmin = Fp16ToF32(b->dmin);
    const int v0 = lane * 8;
    const int gg = v0 >> 6;
    const int half = (v0 >> 5) & 1;
    const int l0 = v0 & 31;
    float s = 0.0F;
    float m = 0.0F;
    QScaleMin(2 * gg + half, b->scales, d, dmin, s, m);
    const uint64_t q8 = LoadU64(b->qs + gg * 32 + l0);
    const uint64_t h8 = LoadU64(b->qh + l0);
    const int sh = 2 * gg + half;
    const uint32_t qw[2] = {static_cast<uint32_t>(q8),
                            static_cast<uint32_t>(q8 >> 32)};
    const uint32_t hw[2] = {static_cast<uint32_t>(h8),
                            static_cast<uint32_t>(h8 >> 32)};
    const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
    const float4 xa = *reinterpret_cast<const float4*>(xv);
    const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
    float acc = 0.0F;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const int b8 = 8 * (p & 3);
      const uint32_t q = (qw[p >> 2] >> b8) & 0xFFU;
      const uint32_t hi = (hw[p >> 2] >> (b8 + sh)) & 1U;
      const int v = static_cast<int>(half ? (q >> 4) : (q & 0x0FU)) +
                    static_cast<int>(hi << 4U);
      const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
      acc += (s * static_cast<float>(v) - m) * xv8;
    }
    return acc;
  }
  const NblQ4_K* b = reinterpret_cast<const NblQ4_K*>(w) + block;
  const float d = Fp16ToF32(b->d);
  const float dmin = Fp16ToF32(b->dmin);
  const int g = lane >> 2;
  float s = 0.0F;
  float m = 0.0F;
  QScaleMin(g, b->scales, d, dmin, s, m);
  const uint64_t q8 = LoadU64(b->qs + (g >> 1) * 32 + (lane & 3) * 8);
  const int sh = (g & 1) * 4;
  const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
  const float4 xa = *reinterpret_cast<const float4*>(xv);
  const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
  const uint32_t qw[2] = {static_cast<uint32_t>(q8),
                          static_cast<uint32_t>(q8 >> 32)};
  float acc = 0.0F;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint32_t byte = qw[p >> 2] >> (8 * (p & 3));
    const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
    acc += (s * static_cast<float>((byte >> sh) & 0x0FU) - m) * xv8;
  }
  return acc;
}

// ---------------------------------------------------------------- variant 5
// Exact replica of the shipped production LoadBytes8 (runtime 8-alignment
// branch) to check whether the branch costs anything versus the v2
// compile-time aligned load.

__device__ __forceinline__ uint2 LoadBytes8Prod(const uint8_t* w8) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(w8);
  if ((a & 7U) == 0U) {
    uint64_t v;
    __builtin_memcpy(&v, w8, sizeof(v));
    return make_uint2(static_cast<uint32_t>(v),
                      static_cast<uint32_t>(v >> 32));
  }
  const uint32_t* const wd =
      reinterpret_cast<const uint32_t*>(a & ~uintptr_t(3));
  const uint32_t sh = static_cast<uint32_t>(a & 3U) * 8U;
  const uint32_t d0 = wd[0];
  const uint32_t d1 = wd[1];
  if (sh == 0U) {
    return make_uint2(d0, d1);
  }
  const uint32_t d2 = wd[2];
  return make_uint2(__funnelshift_r(d0, d1, sh), __funnelshift_r(d1, d2, sh));
}

template<int TYPE>
__device__ __forceinline__ float WarpDotV5(const uint8_t* w, const float* x,
                                           int block, int lane) {
  if (TYPE == kQ5) {
    const NblQ5_K* b = reinterpret_cast<const NblQ5_K*>(w) + block;
    const float d = Fp16ToF32(b->d);
    const float dmin = Fp16ToF32(b->dmin);
    const int v0 = lane * 8;
    const int gg = v0 >> 6;
    const int half = (v0 >> 5) & 1;
    const int l0 = v0 & 31;
    float s = 0.0F;
    float m = 0.0F;
    QScaleMin(2 * gg + half, b->scales, d, dmin, s, m);
    const uint2 q8 = LoadBytes8Prod(b->qs + gg * 32 + l0);
    const uint2 h8 = LoadBytes8Prod(b->qh + l0);
    const int sh = 2 * gg + half;
    const uint32_t qw[2] = {q8.x, q8.y};
    const uint32_t hw[2] = {h8.x, h8.y};
    const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
    const float4 xa = *reinterpret_cast<const float4*>(xv);
    const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
    float acc = 0.0F;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const int b8 = 8 * (p & 3);
      const uint32_t q = (qw[p >> 2] >> b8) & 0xFFU;
      const uint32_t hi = (hw[p >> 2] >> (b8 + sh)) & 1U;
      const int v = static_cast<int>(half ? (q >> 4) : (q & 0x0FU)) +
                    static_cast<int>(hi << 4U);
      const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
      acc += (s * static_cast<float>(v) - m) * xv8;
    }
    return acc;
  }
  const NblQ4_K* b = reinterpret_cast<const NblQ4_K*>(w) + block;
  const float d = Fp16ToF32(b->d);
  const float dmin = Fp16ToF32(b->dmin);
  const int g = lane >> 2;
  float s = 0.0F;
  float m = 0.0F;
  QScaleMin(g, b->scales, d, dmin, s, m);
  const uint2 q8 = LoadBytes8Prod(b->qs + (g >> 1) * 32 + (lane & 3) * 8);
  const int sh = (g & 1) * 4;
  const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
  const float4 xa = *reinterpret_cast<const float4*>(xv);
  const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
  const uint32_t qw[2] = {q8.x, q8.y};
  float acc = 0.0F;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint32_t byte = qw[p >> 2] >> (8 * (p & 3));
    const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
    acc += (s * static_cast<float>((byte >> sh) & 0x0FU) - m) * xv8;
  }
  return acc;
}

// ---------------------------------------------------------------- variant 3

template<int TYPE>
__device__ __forceinline__ float WarpDotV3(const uint8_t* w, const float* x,
                                           int block, int lane) {
  if (TYPE == kQ6) {
    const NblQ6_K* b = reinterpret_cast<const NblQ6_K*>(w) + block;
    const int c = lane >> 4;
    const int k = (lane >> 2) & 3;
    const int hi16 = (lane >> 1) & 1;
    const int j0 = (lane & 3) * 8;
    const uint64_t h8 = LoadBytes8U64(b->qh + c * 32 + j0);
    const uint64_t q8 = LoadBytes8U64(b->ql + c * 64 + (k & 1) * 32 + j0);
    const float ds =
        Fp16ToF32(b->d) * static_cast<float>(b->scales[c * 8 + 2 * k + hi16]);
    const int sh_h = 2 * k;
    const int sh_q = (k >> 1) * 4;
    const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
    const float4 xa = *reinterpret_cast<const float4*>(xv);
    const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
    const uint32_t hw[2] = {static_cast<uint32_t>(h8),
                            static_cast<uint32_t>(h8 >> 32)};
    const uint32_t qw[2] = {static_cast<uint32_t>(q8),
                            static_cast<uint32_t>(q8 >> 32)};
    // u = (h<<4) + lo in [0,63]; sum(u*x) then subtract 32*sum(x), one
    // fma per value and no quarter-rate conversions.
    float acc = 0.0F;
    float sx = 0.0F;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint32_t hbyte = (hw[p >> 2] >> (8 * (p & 3) + sh_h)) & 0x3U;
      const uint32_t lo = (qw[p >> 2] >> (8 * (p & 3) + sh_q)) & 0xFU;
      const float u = U32ToF32Magic((hbyte << 4) | lo);
      const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
      acc = __fmaf_rn(u, xv8, acc);
      sx += xv8;
    }
    // sum (u-32)*x with separate accumulation; ds outside.
    return ds * (acc - 32.0F * sx);
  }
  if (TYPE == kQ5) {
    const NblQ5_K* b = reinterpret_cast<const NblQ5_K*>(w) + block;
    const float d = Fp16ToF32(b->d);
    const float dmin = Fp16ToF32(b->dmin);
    const int v0 = lane * 8;
    const int gg = v0 >> 6;
    const int half = (v0 >> 5) & 1;
    const int l0 = v0 & 31;
    float s = 0.0F;
    float m = 0.0F;
    QScaleMin(2 * gg + half, b->scales, d, dmin, s, m);
    const uint64_t q8 = LoadU64(b->qs + gg * 32 + l0);
    const uint64_t h8 = LoadU64(b->qh + l0);
    const int sh = 2 * gg + half;
    const uint32_t qw[2] = {static_cast<uint32_t>(q8),
                            static_cast<uint32_t>(q8 >> 32)};
    const uint32_t hw[2] = {static_cast<uint32_t>(h8),
                            static_cast<uint32_t>(h8 >> 32)};
    const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
    const float4 xa = *reinterpret_cast<const float4*>(xv);
    const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
    float acc = 0.0F;
    float mx = 0.0F;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const int b8 = 8 * (p & 3);
      const uint32_t q = (qw[p >> 2] >> b8) & 0xFFU;
      const uint32_t hi = (hw[p >> 2] >> (b8 + sh)) & 1U;
      const uint32_t v = (half ? (q >> 4) : (q & 0x0FU)) | (hi << 4U);
      const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
      acc = __fmaf_rn(U32ToF32Magic(v), xv8, acc);
      mx += xv8;
    }
    // s*sum(v*x) - m*sum(x): one mul per super-block, not per value.
    return s * acc - m * mx;
  }
  const NblQ4_K* b = reinterpret_cast<const NblQ4_K*>(w) + block;
  const float d = Fp16ToF32(b->d);
  const float dmin = Fp16ToF32(b->dmin);
  const int g = lane >> 2;
  float s = 0.0F;
  float m = 0.0F;
  QScaleMin(g, b->scales, d, dmin, s, m);
  const uint64_t q8 = LoadU64(b->qs + (g >> 1) * 32 + (lane & 3) * 8);
  const int sh = (g & 1) * 4;
  const float* xv = x + static_cast<int64_t>(block) * kKqElems + lane * 8;
  const float4 xa = *reinterpret_cast<const float4*>(xv);
  const float4 xb = *reinterpret_cast<const float4*>(xv + 4);
  const uint32_t qw[2] = {static_cast<uint32_t>(q8),
                          static_cast<uint32_t>(q8 >> 32)};
  float acc = 0.0F;
  float mx = 0.0F;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    const uint32_t nib = (qw[p >> 2] >> (8 * (p & 3) + sh)) & 0xFU;
    const float xv8 = p < 4 ? (&xa.x)[p] : (&xb.x)[p - 4];
    acc = __fmaf_rn(U32ToF32Magic(nib), xv8, acc);
    mx += xv8;
  }
  return s * acc - m * mx;
}

// ---------------------------------------------------------------- kernels

template<int TYPE>
__device__ __forceinline__ uint64_t WarpReadVRead(const uint8_t* w, int block,
                                                  int lane) {
  // Touch exactly the bytes the dot reads, sum cheaply to defeat DCE.
  const int stride = (TYPE == kQ6) ? 210 : (TYPE == kQ5) ? 176 : 144;
  const uint8_t* b = w + static_cast<int64_t>(block) * stride;
  uint64_t s = 0;
  if (TYPE == kQ6) {
    const int c = lane >> 4;
    const int k = (lane >> 2) & 3;
    const int j0 = (lane & 3) * 8;
    s += LoadBytes8U64(b + 128 + c * 32 + j0);
    s += LoadBytes8U64(b + c * 64 + (k & 1) * 32 + j0);
    s += b[192 + (c * 8 + 2 * k + ((lane >> 1) & 1))];
  } else if (TYPE == kQ5) {
    const int l0 = (lane * 8) & 31;
    s += LoadU64(b + 48 + l0);
    s += LoadU64(b + 16 + l0);
    s += b[4 + (lane * 8 >> 6)];  // scales byte of sub-quant 2gg+half
  } else {
    const int g = lane >> 2;
    s += LoadU64(b + 16 + (g >> 1) * 32 + (lane & 3) * 8);
    s += b[4 + (g >> 1)];
  }
  return s;
}

template<int TYPE, int VAR>
__global__ void KqGemvKernel(const float* __restrict__ x,
                             const uint8_t* __restrict__ w,
                             float* __restrict__ out, int n_out, int k,
                             int stride, int warps_per_row,
                             int blocks_per_slot) {
  const int warp = static_cast<int>(threadIdx.x) / warpSize;
  const int lane = static_cast<int>(threadIdx.x) % warpSize;
  const int row_local = warp / warps_per_row;
  const int warp_in_row = warp % warps_per_row;
  __shared__ float fold[64];
  if (static_cast<int>(threadIdx.x) < 64) {
    fold[threadIdx.x] = 0.0F;
  }
  __syncthreads();
  const int row =
      static_cast<int>(blockIdx.x) * (kBlock / warpSize / warps_per_row) +
      row_local;
  const bool active = row < n_out;
  float acc = 0.0F;
  if (active) {
    const uint8_t* wr = w + static_cast<int64_t>(row) * stride;
    const float* xr = x;
    const int n_super = k / kKqElems;
    if constexpr (VAR == 0 || VAR == 2 || VAR == 3 || VAR == 5) {
      for (int s = warp_in_row; s < n_super; s += warps_per_row) {
        if constexpr (VAR == 0) {
          acc += WarpDotV0<TYPE>(wr, xr, s, lane);
        } else if constexpr (VAR == 2) {
          acc += WarpDotV2<TYPE>(wr, xr, s, lane);
        } else if constexpr (VAR == 5) {
          acc += WarpDotV5<TYPE>(wr, xr, s, lane);
        } else {
          acc += WarpDotV3<TYPE>(wr, xr, s, lane);
        }
      }
    } else if constexpr (VAR == 1 || VAR == 4) {
      int s = warp_in_row;
      for (; s + warps_per_row < n_super; s += 2 * warps_per_row) {
        float a0;
        float a1;
        if constexpr (VAR == 1) {
          a0 = WarpDotV0<TYPE>(wr, xr, s, lane);
          a1 = WarpDotV0<TYPE>(wr, xr, s + warps_per_row, lane);
        } else {
          a0 = WarpDotV3<TYPE>(wr, xr, s, lane);
          a1 = WarpDotV3<TYPE>(wr, xr, s + warps_per_row, lane);
        }
        acc += a0;
        acc += a1;
      }
      for (; s < n_super; s += warps_per_row) {
        acc += (VAR == 1) ? WarpDotV0<TYPE>(wr, xr, s, lane)
                          : WarpDotV3<TYPE>(wr, xr, s, lane);
      }
    } else {
      uint64_t r = 0;
      for (int bs = warp_in_row; bs < n_super; bs += warps_per_row) {
        r += WarpReadVRead<TYPE>(wr, bs, lane);
      }
      acc = static_cast<float>(r & 255U);
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(~0ULL, acc, off);
  }
  if (lane == 0) {
    fold[row_local * 8 + warp_in_row] = acc;
  }
  __syncthreads();
  if (active && lane == 0 && warp_in_row == 0) {
    float sum = 0.0F;
    for (int j = 0; j < kBlock / warpSize; ++j) {
      sum += fold[row_local * 8 + j];
    }
    out[row] = sum;
  }
}

// ---------------------------------------------------------------- harness

float RandUnit() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5F;
}

struct Shape {
  const char* name;
  int type;   // Tier
  int n_out;
  int k;
};

constexpr Shape kShapes[] = {
    {"ssm_in  18560x10240 Q6", kQ6, 18560, 10240},
    {"ssm_out 10240x8192  Q6", kQ6, 10240, 8192},
    {"ud_exp_up   1024x2048 Q6", kQ6, 1024, 2048},
    {"ud_exp_down 2048x1024 Q6", kQ6, 2048, 1024},
    {"up_shexp 5376x10240 Q5", kQ5, 5376, 10240},
    {"qkv     5376x10240  Q4", kQ4, 5376, 10240},
};

template<int TYPE, int VAR>
void LaunchOne(const float* x, const uint8_t* w, float* out, int n_out, int k,
               int stride, hipStream_t stream) {
  const int n_chunk = k / kKqElems;
  const int warps_per_row = (n_chunk + kWarp - 1) / kWarp;
  const int rows_per_block = kBlock / (kWarp * warps_per_row);
  const int blocks_per_slot = (n_out + rows_per_block - 1) / rows_per_block;
  KqGemvKernel<TYPE, VAR><<<blocks_per_slot, kBlock, 0, stream>>>(
      x, w, out, n_out, k, stride, warps_per_row, blocks_per_slot);
}

template<int TYPE>
float RunShape(const Shape& sh) {
  const int stride = (TYPE == kQ6) ? 210 : (TYPE == kQ5) ? 176 : 144;
  const size_t wbytes = static_cast<size_t>(sh.n_out) * stride * (sh.k / 256);
  uint8_t* w = nullptr;
  float* x = nullptr;
  float* out0 = nullptr;
  float* outv = nullptr;
  CHECK(hipMalloc(&w, wbytes));
  CHECK(hipMalloc(&x, static_cast<size_t>(sh.k) * sizeof(float)));
  CHECK(hipMalloc(&out0, static_cast<size_t>(sh.n_out) * sizeof(float)));
  CHECK(hipMalloc(&outv, static_cast<size_t>(sh.n_out) * sizeof(float)));
  std::vector<uint8_t> hw(wbytes);
  for (size_t i = 0; i < wbytes; ++i) {
    hw[i] = static_cast<uint8_t>(rand() & 0xFF);
  }
  // Keep scales/d finite and moderate so the reference sum stays sane.
  for (size_t blk = 0; blk < wbytes; blk += stride) {
    if (TYPE == kQ6) {
      for (int j = 0; j < 16; ++j) hw[blk + 192 + j] = 64 + (j % 7) - 3;
      hw[blk + 208] = 0x77;
      hw[blk + 209] = 0x3D;  // ~0.06
    } else {
      hw[blk + 0] = 0x77;
      hw[blk + 1] = 0x3D;
      hw[blk + 2] = 0x77;
      hw[blk + 3] = 0x3D;
    }
  }
  std::vector<float> hx(sh.k);
  for (auto& v : hx) v = RandUnit();
  CHECK(hipMemcpy(w, hw.data(), wbytes, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(x, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t ev0;
  hipEvent_t ev1;
  CHECK(hipEventCreate(&ev0));
  CHECK(hipEventCreate(&ev1));
  float best = 1e30F;
  float base_ms = 0.0F;
  const int reps = 11;
  auto timed = [&](auto launch) {
    for (int i = 0; i < 3; ++i) launch();
    CHECK(hipDeviceSynchronize());
    float tot = 0.0F;
    for (int i = 0; i < reps; ++i) {
      CHECK(hipEventRecord(ev0));
      launch();
      CHECK(hipEventRecord(ev1));
      CHECK(hipEventSynchronize(ev1));
      float ms = 0.0F;
      CHECK(hipEventElapsedTime(&ms, ev0, ev1));
      tot += ms;
    }
    return tot / reps;
  };
  base_ms = timed([&] { LaunchOne<TYPE, 0>(x, w, out0, sh.n_out, sh.k, stride, 0); });
  printf("  %-26s v0 %8.3f ms %7.1f GB/s\n", sh.name, base_ms,
         wbytes / (base_ms * 1e6F));

  auto check_v = [&](auto launch, const char* tag) {
    CHECK(hipMemsetAsync(outv, 0, static_cast<size_t>(sh.n_out) * sizeof(float)));
    launch();
    CHECK(hipDeviceSynchronize());
    std::vector<float> a(sh.n_out);
    std::vector<float> b(sh.n_out);
    CHECK(hipMemcpy(a.data(), out0, a.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(b.data(), outv, b.size() * sizeof(float), hipMemcpyDeviceToHost));
    double worst = 0.0;
    for (int i = 0; i < sh.n_out; ++i) {
      const double denom = std::fabs(a[i]) + 1e-2;
      worst = std::max(worst, std::fabs(a[i] - b[i]) / denom);
    }
    printf("  %-26s correctness %s rel=%.2e%s\n", "", tag, worst,
           (worst < 1e-3) ? "" : "  <-- TOO LARGE");
  };

#define RUNVAR(V, tag)                                                    \
  {                                                                       \
    float ms = 0.0F;                                                      \
    if constexpr ((V) == 99) {                                            \
      ms = timed([&] { LaunchOne<TYPE, 99>(x, w, outv, sh.n_out, sh.k,  \
                                           stride, 0); });              \
      printf("  %-26s %s %8.3f ms %7.1f GB/s (roof)\n", sh.name, tag,   \
             ms, (wbytes) / (ms * 1e6F));                                 \
    } else {                                                              \
      ms = timed([&] { LaunchOne<TYPE, (V)>(x, w, outv, sh.n_out,       \
                                            sh.k, stride, 0); });       \
      printf("  %-26s %s %8.3f ms %7.1f GB/s (%.3fx)\n", sh.name, tag,  \
             ms, (wbytes) / (ms * 1e6F), base_ms / ms);                 \
    }                                                                     \
  }
  RUNVAR(99, "vread");
  RUNVAR(1, "v1 ");
  RUNVAR(2, "v2 ");
  RUNVAR(3, "v3 ");
  RUNVAR(4, "v4 ");
  if constexpr (TYPE != kQ6) {
    RUNVAR(5, "v5 ");  // shipped production LoadBytes8 (runtime align check)
  }
  check_v([&] { LaunchOne<TYPE, 1>(x, w, outv, sh.n_out, sh.k, stride, 0); }, "v1");
  check_v([&] { LaunchOne<TYPE, 2>(x, w, outv, sh.n_out, sh.k, stride, 0); }, "v2");
  check_v([&] { LaunchOne<TYPE, 3>(x, w, outv, sh.n_out, sh.k, stride, 0); }, "v3");
  check_v([&] { LaunchOne<TYPE, 4>(x, w, outv, sh.n_out, sh.k, stride, 0); }, "v4");
  if constexpr (TYPE != kQ6) {
    check_v([&] { LaunchOne<TYPE, 5>(x, w, outv, sh.n_out, sh.k, stride, 0); }, "v5");
  }
  CHECK(hipEventDestroy(ev0));
  CHECK(hipEventDestroy(ev1));
  CHECK(hipFree(w));
  CHECK(hipFree(x));
  CHECK(hipFree(out0));
  CHECK(hipFree(outv));
  return best;
}

}  // namespace

int main() {
  srand(1234);
  int dev = 0;
  CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, dev));
  printf("device: %s\n", prop.name);
  RunShape<kQ6>(kShapes[0]);
  RunShape<kQ6>(kShapes[1]);
  RunShape<kQ6>(kShapes[2]);
  RunShape<kQ6>(kShapes[3]);
  RunShape<kQ5>(kShapes[4]);
  RunShape<kQ4>(kShapes[5]);
  return 0;
}
