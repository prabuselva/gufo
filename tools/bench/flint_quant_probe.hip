// C7u activation-quantize structure probe. The shipped
// QuantizeActivationToQ8_1Kernel runs one wavefront per 32-element Q8_1 block
// (one element per thread): at pp8192 an average call reads 33-67 MB at only
// ~95 GB/s against the ~239 GB/s measured copy ceiling (tools/prof pp8192:
// 588 calls, 263.66 ms, mean 448 us). This probe times a variant where one
// wavefront owns four blocks through a float4 per lane: the block belongs to
// 8 consecutive lanes, so the per-block max is a group-of-8 fmaxf butterfly.
// max, int sums and the per-element d/id/roundf math are exact under any
// evaluation order, so every stored byte, scale and sidecar value must be
// bit-identical to the shipped kernel; the probe verifies that over the whole
// tiled payload plus sidecar and checks the untouched tail bytes.
//
// Build:
//   PATH=/opt/rocm/lib/llvm/bin:$PATH clang++ --offload-arch=gfx1151 \
//     --driver-mode=g++ -L/opt/rocm/lib --hip-link -O3 -std=c++20 -x hip \
//     -I . tools/bench/flint_quant_probe.hip -o /tmp/flint_quant_probe
#include <chrono>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"

namespace probe {

using gufo::hip::kQ8ActScaleOffset;
using gufo::hip::Q8ActSumBytes;
using gufo::hip::kQ8ActTileBytes;
using gufo::hip::kQ8ActTileTokens;
using gufo::hip::Q8ActTiledBytes;

// Verbatim copy of the shipped QuantizeActivationToQ8_1Kernel<true, float>
// (prefill_quant_gemm.hip).
__global__ void ShippedKernel(const float* __restrict__ x,
                              void* __restrict__ y, std::size_t batch,
                              std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;

  const std::size_t total_blocks = batch * num_blocks;
  if (b_idx >= total_blocks) {
    return;
  }

  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const std::size_t elem_idx = (tok * k) + (blk * 32) + lane_id;

  const float val = x[elem_idx];
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }

  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const std::int8_t q = static_cast<std::int8_t>(roundf(val * id));

  gufo::hip::StoreQ8ActLane<true>(y, batch, num_blocks, tok, blk, lane_id, d,
                                  q);
}

// One wavefront owns 4 consecutive blocks (128 elements) of one row. Element
// 4*l+j of the chunk lives in block group=l/8 at position 4*(l&7)+j, so each
// block is owned by exactly 8 consecutive lanes and needs only a 3-step
// group-of-8 butterfly. Exact reductions -> bit-identical to the shipped
// kernel. Requires num_blocks % 4 == 0 (k % 128 == 0), which the launcher
// falls back on otherwise.
__global__ void Float4Kernel(const float4* __restrict__ x4, void* __restrict__ y,
                             std::size_t batch, std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t blocks4 = num_blocks / 4;
  const std::size_t wave = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
  const std::size_t lane = threadIdx.x & 31u;
  const std::size_t total = batch * blocks4;
  if (wave >= total) {
    return;
  }
  const std::size_t tok = wave / blocks4;
  const std::size_t g = wave % blocks4;
  const std::size_t grp = lane >> 3u;      // block within the chunk
  const std::size_t sub = lane & 7u;       // lane within the block
  const std::size_t blk = g * 4u + grp;
  const std::size_t tl = tok % kQ8ActTileTokens;
  const std::size_t tt = tok / kQ8ActTileTokens;

  const float4 v = x4[(tok * k + g * 128u) / 4u + lane];
  float comp[4] = {fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w)};
  float mx = fmaxf(fmaxf(comp[0], comp[1]), fmaxf(comp[2], comp[3]));
  for (int off = 4; off > 0; off >>= 1) {
    mx = fmaxf(mx, __shfl_xor(mx, off));
  }
  const float d = mx / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;

  std::int8_t* tile =
      static_cast<std::int8_t*>(y) + ((tt * num_blocks + blk) * kQ8ActTileBytes);
  int qsum = 0;
  const std::size_t p0 = sub * 4u;
  const float elems[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (std::size_t j = 0; j < 4u; ++j) {
    const std::size_t pos = p0 + j;
    const std::int8_t q = static_cast<std::int8_t>(roundf(elems[j] * id));
    tile[((pos >> 4u) * 256u) + (tl * 16u) + (pos & 15u)] = q;
    qsum += static_cast<int>(q);
    if (sub == 0u && j == 0u) {
      *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + tl * sizeof(float)) =
          d;
    }
  }
  for (int off = 4; off > 0; off >>= 1) {
    qsum += __shfl_xor(qsum, off);
  }
  if (sub == 0u) {
    float* sums = reinterpret_cast<float*>(static_cast<std::uint8_t*>(y) +
                                           Q8ActTiledBytes(batch, k));
    sums[((tt * num_blocks + blk) * kQ8ActTileTokens) + tl] =
        d * static_cast<float>(qsum);
  }
}

// Read-all calibration: stream x with float4, block-reduce into a dummy sum.
extern "C" __global__ void ReadCalib(const float4* __restrict__ x4,
                                     std::size_t count, float* __restrict__ out) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = blockDim.x * gridDim.x;
  float acc = 0.0F;
  for (std::size_t j = i; j < count; j += stride) {
    const float4 v = x4[j];
    acc += (v.x + v.y) + (v.z + v.w);
  }
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_xor(acc, off);
  }
  if ((threadIdx.x & 31u) == 0u) {
    atomicAdd(out, acc);
  }
}

template <typename Fn>
double TimeKernel(Fn launch, int reps) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  launch();
  hipDeviceSynchronize();
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    hipEventRecord(e0);
    launch();
    hipEventRecord(e1);
    hipEventSynchronize(e1);
    float ms = 0.0F;
    hipEventElapsedTime(&ms, e0, e1);
    if (ms < best) {
      best = ms;
    }
  }
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  return best;
}

struct Shape {
  const char* name;
  std::size_t batch, k;
};

void RunShape(const Shape& sh) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-8.0F, 8.0F);
  const std::size_t elems = sh.batch * sh.k;
  std::vector<float> host(elems);
  for (auto& v : host) {
    v = dist(rng);
  }
  // Exercise the d == 0 path and saturation edges in a full block.
  for (std::size_t e = 0; e < 32; ++e) {
    host[sh.k + e] = 0.0F;
  }
  host[2 * sh.k + 5] = -0.0F;
  host[2 * sh.k + 7] = 1e30F;
  host[2 * sh.k + 9] = -1e30F;

  const std::size_t num_blocks = sh.k / 32;
  const std::size_t out_bytes =
      Q8ActTiledBytes(sh.batch, sh.k) + Q8ActSumBytes(sh.batch, sh.k);
  const std::size_t guard = 4096;

  float* dx;
  void *dy_a, *dy_b;
  hipMalloc(&dx, elems * sizeof(float));
  hipMalloc(&dy_a, out_bytes + guard);
  hipMalloc(&dy_b, out_bytes + guard);
  hipMemcpy(dx, host.data(), elems * sizeof(float), hipMemcpyHostToDevice);
  hipMemset(dy_a, 0xA5, out_bytes + guard);
  hipMemset(dy_b, 0xA5, out_bytes + guard);

  auto launch_shipped = [&] {
    ShippedKernel<<<(num_blocks * sh.batch + 7) / 8, 256>>>(dx, dy_a, sh.batch,
                                                            sh.k);
  };
  auto launch_float4 = [&] {
    Float4Kernel<<<(sh.batch * num_blocks / 4 + 7) / 8, 256>>>(
        reinterpret_cast<const float4*>(dx), dy_b, sh.batch, sh.k);
  };
  launch_shipped();
  launch_float4();
  hipDeviceSynchronize();
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    printf("%s: launch error %s\n", sh.name, hipGetErrorString(err));
    return;
  }

  std::vector<std::uint8_t> a(out_bytes + guard), b(out_bytes + guard);
  hipMemcpy(a.data(), dy_a, out_bytes + guard, hipMemcpyDeviceToHost);
  hipMemcpy(b.data(), dy_b, out_bytes + guard, hipMemcpyDeviceToHost);
  std::size_t mism = 0;
  for (std::size_t i = 0; i < out_bytes; ++i) {
    if (a[i] != b[i]) {
      if (mism < 5) {
        printf("%s: mismatch at byte %zu (%02x vs %02x)\n", sh.name, i, a[i],
               b[i]);
      }
      ++mism;
    }
  }
  for (std::size_t i = out_bytes; i < out_bytes + guard; ++i) {
    if (a[i] != 0xA5 || b[i] != 0xA5) {
      printf("%s: OOB write at %zu\n", sh.name, i);
      ++mism;
    }
  }

  const double ta = TimeKernel(launch_shipped, 10);
  const double tb = TimeKernel(launch_float4, 10);
  // Traffic: fp32 read + q8/scale/sum write.
  const double bytes = elems * sizeof(float) + Q8ActTiledBytes(sh.batch, sh.k) +
                       Q8ActSumBytes(sh.batch, sh.k);
   printf(
       "%s B=%zu k=%zu  m=%-6zu shipped %.3f ms  float4 %.3f ms  (%+.1f %%) "
       "gb/s %.0f -> %.0f\n",
       sh.name, sh.batch, sh.k, mism, ta, tb, (tb / ta - 1.0) * 100.0,
       bytes / (ta * 1e6), bytes / (tb * 1e6));

  hipFree(dx);
  hipFree(dy_a);
  hipFree(dy_b);
}

}  // namespace probe

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  // Read ceiling calibration at the dominant pp8192 activation size.
  {
    const std::size_t elems = 4096u * 2048u;
    float* d;
    float* dout;
    hipMalloc(&d, elems * 4);
    hipMalloc(&dout, 4);
    hipMemset(d, 0x3c, elems * 4);
    std::size_t count = elems / 4;
    auto calib = [&] { probe::ReadCalib<<<2048, 256>>>(reinterpret_cast<const float4*>(d), count, dout); };
    double t = probe::TimeKernel(calib, 10);
    printf("read-calib 4096x2048 fp32: %.3f ms -> %.0f GB/s\n", t,
           elems * 4.0 / (t * 1e6));
    hipFree(d);
    hipFree(dout);
  }
  probe::RunShape({"h", 4096, 2048});
  probe::RunShape({"qd", 4096, 4096});
  probe::RunShape({"ff", 4096, 1024});
  probe::RunShape({"mid", 2048, 2048});
  probe::RunShape({"tail", 1000, 256});  // batch not multiple of the tile
  probe::RunShape({"b4edge", 64, 128});  // smallest legal W8A8 shape scale
  return 0;
}
