// C15 probe: down-epilogue fused atomic accumulation vs the shipped
// "store bf16 slot rows + SumExperts" data-movement contract at Flint
// pp2048 shapes (B=2048, H=2048, top-6 of 256 experts).
//
// baseline = K1 scattered bf16 slot store (the down epilogue today)
//            + K2 SumExperts streaming sum (shipped fast path).
// fused    = K3 expert-sorted 6-way f32 atomicAdd epilogue (the proposal)
//            + K4 shared-expert add (SumExperts minus the routed read).
#include <hip/hip_fp16.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

namespace {

constexpr int kB = 2048;
constexpr int kH = 2048;
constexpr int kUse = 6;
constexpr int kExperts = 256;
constexpr int kColTile = 128;      // down grid.x granularity
constexpr int kTileRows = 128;     // 128-row tiles

#define CHECK(call)                                                     \
  do {                                                                  \
    hipError_t e = (call);                                              \
    if (e != hipSuccess) {                                              \
      std::fprintf(stderr, "hip error %s @%d\n", hipGetErrorString(e),  \
                   __LINE__);                                           \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

// K1: the down epilogue store today - one full row of bf16 per compact
// slot, destination-addressed like production (token-major ed_h rows hit
// in expert-sorted order).
__global__ void StoreHalfKernel(__half* __restrict__ ed,
                                const int* __restrict__ dst, int rows) {
  const int j = blockIdx.x;
  if (j >= rows) {
    return;
  }
  __half* o = ed + static_cast<std::size_t>(dst[j]) * kH;
  union {
    uint4 u;
    __half h[8];
  } v;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    v.h[q] = __float2half(static_cast<float>((j % 97) - 48 + q));
  }
  for (int c = threadIdx.x; c < kH / 8; c += blockDim.x) {
    *reinterpret_cast<uint4*>(o + c * 8) = v.u;
  }
}

// K2: shipped SumExperts half group path (weights hoisted, 16-byte rows).
__global__ void SumExpertsKernel(float* __restrict__ x,
                                 const __half* __restrict__ routed,
                                 const float* __restrict__ weights,
                                 const float* __restrict__ shared) {
  const int t = blockIdx.y;
  const float* wrow = weights + static_cast<std::size_t>(t) * kUse;
  float wu[kUse];
#pragma unroll
  for (int s = 0; s < kUse; ++s) {
    wu[s] = wrow[s];
  }
  const std::size_t cols8 = kH / 8;
  const std::size_t row8 = static_cast<std::size_t>(t) * kUse * cols8;
  for (std::size_t c = threadIdx.x; c < cols8; c += blockDim.x) {
    const std::size_t i = row8 + c;
    float4 acc[2] = {make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0)};
#pragma unroll
    for (int s = 0; s < kUse; ++s) {
      union {
        uint4 u;
        __half2 h[4];
      } v;
      v.u = *reinterpret_cast<const uint4*>(routed +
                                           (i + static_cast<std::size_t>(s) * cols8) * 8);
      acc[0].x += wu[s] * __half2float(v.h[0].x);
      acc[0].y += wu[s] * __half2float(v.h[0].y);
      acc[0].z += wu[s] * __half2float(v.h[1].x);
      acc[0].w += wu[s] * __half2float(v.h[1].y);
      acc[1].x += wu[s] * __half2float(v.h[2].x);
      acc[1].y += wu[s] * __half2float(v.h[2].y);
      acc[1].z += wu[s] * __half2float(v.h[3].x);
      acc[1].w += wu[s] * __half2float(v.h[3].y);
    }
    float4* xv = reinterpret_cast<float4*>(x + static_cast<std::size_t>(t) *
                                                        kH) +
                 2 * (i - row8);
    float4 a0 = xv[0];
    float4 a1 = xv[1];
    const float4 sh =
        reinterpret_cast<const float4*>(shared + static_cast<std::size_t>(t) *
                                                    kH)[2 * (i - row8)];
    const float4 sh1 =
        reinterpret_cast<const float4*>(shared + static_cast<std::size_t>(t) *
                                                    kH)[2 * (i - row8) + 1];
    a0.x += acc[0].x + sh.x;
    a0.y += acc[0].y + sh.y;
    a0.z += acc[0].z + sh.z;
    a0.w += acc[0].w + sh.w;
    a1.x += acc[1].x + sh1.x;
    a1.y += acc[1].y + sh1.y;
    a1.z += acc[1].z + sh1.z;
    a1.w += acc[1].w + sh1.w;
    xv[0] = a0;
    xv[1] = a1;
  }
}

// K3: the proposal - same tile/epilogue walk as the down kernel
// (grid.x = 128-col blocks, grid.y = expert tiles, 16 threads per token
// row x 8 columns), but instead of storing bf16 slot rows each thread
// does 8 f32 atomicAdds of weight * value into x at the token row.
__global__ void FusedAtomicKernel(float* __restrict__ x,
                                  const int* __restrict__ tiles,
                                  const int* __restrict__ off,
                                  const int* __restrict__ cnt,
                                  const int* __restrict__ row_token,
                                  const int* __restrict__ row_slot,
                                  const float* __restrict__ weights) {
  const int tile = tiles[blockIdx.y];
  const int e = tile & 0xFFFF;
  const int tt = tile >> 16;
  const int begin = off[e] + tt * kTileRows;
  const int bucket = std::min(cnt[e] - tt * kTileRows, kTileRows);
  if (bucket <= 0) {
    return;
  }
  const int row0 = blockIdx.x * kColTile;
  const int ttok = threadIdx.x / 16;
  const int rloc = (threadIdx.x % 16) * 8;
  const int ntok = std::min(bucket, kTileRows);
  for (int j = 0; j < (ntok + 15) / 16; ++j) {
    const int rt = j * 16 + ttok;
    if (rt < bucket) {
      const int jrow = begin + rt;
      const int tok = row_token[jrow];
      const float w = weights[row_slot[jrow]];
      const float v = w * static_cast<float>((jrow % 101) - 50);
      float* o = x + static_cast<std::size_t>(tok) * kH + row0 + rloc;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        atomicAdd(o + q, v);
      }
    }
    __syncthreads();
  }
}

// K4: shared-expert add only (what SumExperts degenerates to).
__global__ void AddSharedKernel(float* __restrict__ x,
                                const float* __restrict__ shared) {
  const int t = blockIdx.y;
  float4* xv =
      reinterpret_cast<float4*>(x + static_cast<std::size_t>(t) * kH);
  const float4* sv =
      reinterpret_cast<const float4*>(shared + static_cast<std::size_t>(t) *
                                                kH);
  for (int i = threadIdx.x; i < kH / 4; i += blockDim.x) {
    float4 a = xv[i];
    const float4 s = sv[i];
    a.x += s.x;
    a.y += s.y;
    a.z += s.z;
    a.w += s.w;
    xv[i] = a;
  }
}

}  // namespace

int main() {
  const int rows = kB * kUse;
  std::vector<std::pair<int, int>> slots;
  slots.reserve(rows);
  for (int i = 0; i < rows; ++i) {
    slots.emplace_back((i * 7) % kExperts, i);  // (expert, token*6+s)
  }
  std::stable_sort(slots.begin(), slots.end());
  std::vector<int> off(kExperts + 1, 0);
  for (const auto& p : slots) {
    ++off[p.first + 1];
  }
  for (int e = 0; e < kExperts; ++e) {
    off[e + 1] += off[e];
  }
  std::vector<int> row_token(rows), row_slot(rows);
  {
    std::vector<int> cur(off.begin(), off.end());
    for (const auto& p : slots) {
      const int j = cur[p.first]++;
      row_token[j] = p.second / kUse;
      row_slot[j] = p.second;
    }
  }
  std::vector<int> tiles;
  for (int e = 0; e < kExperts; ++e) {
    const int padded = (off[e + 1] - off[e] + 15) / 16 * 16;
    for (int j = 0; j < padded; j += kTileRows) {
      tiles.push_back(e | ((j / kTileRows) << 16));
    }
  }
  std::printf("rows=%d tiles=%zu (capacity order: expert-sorted)\n", rows,
              tiles.size());

  float *x, *x2, *sv, *wts;
  __half* ed;
  int *tiles_d, *off_d, *cnt_d, *rt_d, *rs_d;
  CHECK(hipMalloc(&x, sizeof(float) * kB * kH));
  CHECK(hipMalloc(&x2, sizeof(float) * kB * kH));
  CHECK(hipMalloc(&sv, sizeof(float) * kB * kH));
  CHECK(hipMalloc(&wts, sizeof(float) * rows));
  CHECK(hipMalloc(&ed, sizeof(__half) * rows * kH));
  CHECK(hipMalloc(&tiles_d, sizeof(int) * tiles.size()));
  CHECK(hipMalloc(&off_d, sizeof(int) * (kExperts + 1)));
  CHECK(hipMalloc(&cnt_d, sizeof(int) * kExperts));
  CHECK(hipMalloc(&rt_d, sizeof(int) * rows));
  CHECK(hipMalloc(&rs_d, sizeof(int) * rows));
  std::vector<int> cnt(kExperts);
  for (int e = 0; e < kExperts; ++e) {
    cnt[e] = off[e + 1] - off[e];
  }
  std::vector<float> hw(rows, 0.25F);
  CHECK(hipMemcpy(tiles_d, tiles.data(), sizeof(int) * tiles.size(),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(off_d, off.data(), sizeof(int) * (kExperts + 1),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(cnt_d, cnt.data(), sizeof(int) * kExperts,
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(rt_d, row_token.data(), sizeof(int) * rows,
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(rs_d, row_slot.data(), sizeof(int) * rows,
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(wts, hw.data(), sizeof(float) * rows,
                  hipMemcpyHostToDevice));
  CHECK(hipMemset(x, 0, sizeof(float) * kB * kH));
  CHECK(hipMemset(x2, 0, sizeof(float) * kB * kH));
  CHECK(hipMemset(sv, 0, sizeof(float) * kB * kH));

  const int iters = 40;
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  float ms = 0.0F;

  auto time_step = [&](const char* name, const std::function<void()>& launch)
      -> float {
    for (int i = 0; i < 5; ++i) {
      launch();
    }
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    for (int i = 0; i < iters; ++i) {
      launch();
    }
    CHECK(hipEventRecord(e1));
    CHECK(hipEventSynchronize(e1));
    float t = 0.0F;
    CHECK(hipEventElapsedTime(&t, e0, e1));
    std::printf("%-28s %8.1f us\n", name, t * 1000.0F / iters);
    return t * 1000.0F / iters;
  };

  const float t_store = time_step("K1 store bf16 rows", [&] {
    StoreHalfKernel<<<rows, 256>>>(ed, rs_d, rows);
  });
  const float t_sum = time_step("K2 SumExperts (shipped)", [&] {
    const dim3 g(1, kB);
    SumExpertsKernel<<<g, 256>>>(x, ed, wts, sv);
  });
  const float t_atomic =
      time_step("K3 fused atomic 6-way", [&] {
        const dim3 g(kH / kColTile, static_cast<unsigned>(tiles.size()));
        FusedAtomicKernel<<<g, 256>>>(x, tiles_d, off_d, cnt_d, rt_d, rs_d,
                                      wts);
      });
  const float t_shared = time_step("K4 add shared only", [&] {
    const dim3 g(1, kB);
    AddSharedKernel<<<g, 256>>>(x, sv);
  });

  std::printf(
      "\nbaseline K1+K2   = %8.1f us\n"
      "fused    K3+K4   = %8.1f us (%.1f %%)\n",
      t_store + t_sum, t_atomic + t_shared,
      100.0F * (t_atomic + t_shared - t_store - t_sum) / (t_store + t_sum));
  return 0;
}
