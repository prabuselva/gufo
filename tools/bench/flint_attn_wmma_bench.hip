// Bitwise + timing sweep of WmmaAttentionKernel launch configs for the
// measured Flint prefill shapes (heads 32, kv 4, head 128, B 2048) on
// gfx1151. One wave per 16-row block fixes kRowBlocks = 8 (K/V staging is
// one 8-half chunk per thread over 256 threads), so the valid configs are
// (kHeads, kQueryBlocks) in {(1,8), (2,4), (4,2), (8,1)}. Every config must
// emit bitwise-identical outputs to the shipped <2, 4>; timing runs the
// windowed (513) and full-causal layer shapes.
//
// Build:
//   clang++ --offload-arch=gfx1151 --driver-mode=g++ -L/opt/rocm/lib \
//     --hip-link -O3 -std=c++20 -x hip -I . \
//     tools/bench/flint_attn_wmma_bench.hip -o /tmp/flint_attn_bench
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/models/flint/kernels/rocm/attention_wmma.hip.cpp"

namespace fr = gufo::models::flint::rocm;

namespace {

constexpr std::uint32_t kRows = 2048;
constexpr std::uint32_t kHeadsN = 32;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kPos0 = 512;
constexpr std::size_t kQn =
    static_cast<std::size_t>(kRows) * kHeadsN * kDim;  // 8.4M floats

#define CHECK(cmd)                                                  \
  do {                                                              \
    const hipError_t err__ = (cmd);                                 \
    if (err__ != hipSuccess) {                                      \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",               \
                   hipGetErrorString(err__), __FILE__, __LINE__);   \
      std::exit(1);                                                 \
    }                                                               \
  } while (0)

fr::WmmaArgs MakeArgs(const float* q, const __half* k, const __half* v,
                      const float* sinks, float* out, std::uint32_t window) {
  fr::WmmaArgs a{};
  a.q = q;
  a.k_cache = k;
  a.v_cache = v;
  a.sinks = sinks;
  a.out = out;
  a.scale = 1.0F / std::sqrt(static_cast<float>(kDim));
  a.rows = kRows;
  a.first_position = kPos0;
  a.key_limit = kPos0 + kRows;
  a.window = window;
  a.heads = kHeadsN;
  a.kv_heads = kKvHeads;
  a.kv_stride = kKvHeads * kDim;
  return a;
}

template<std::uint32_t kH, std::uint32_t kQ>
void LaunchCfg(const fr::WmmaArgs& a) {
  fr::WmmaAttentionKernel<kH, kQ>
      <<<dim3((kRows + kQ * 16 - 1) / (kQ * 16), kHeadsN / kH),
         32 * kH * kQ>>>(a);
}

double TimeCfg(int cfg, const fr::WmmaArgs& a, int iters) {
  const auto launch = [&] {
    switch (cfg) {
      case 0: LaunchCfg<1, 8>(a); break;
      case 1: LaunchCfg<2, 4>(a); break;
      case 2: LaunchCfg<4, 2>(a); break;
      default: LaunchCfg<8, 1>(a); break;
    }
  };
  for (int i = 0; i < 3; ++i) launch();
  CHECK(hipDeviceSynchronize());
  hipEvent_t beg, end;
  CHECK(hipEventCreate(&beg));
  CHECK(hipEventCreate(&end));
  CHECK(hipEventRecord(beg));
  for (int i = 0; i < iters; ++i) launch();
  CHECK(hipEventRecord(end));
  CHECK(hipEventSynchronize(end));
  float ms = 0.0F;
  CHECK(hipEventElapsedTime(&ms, beg, end));
  CHECK(hipEventDestroy(beg));
  CHECK(hipEventDestroy(end));
  return ms / iters;
}

}  // namespace

int main() {
  std::vector<float> h_q(kQn);
  std::vector<float> h_sink(kHeadsN, 0.75F);
  const std::size_t kcn =
      static_cast<std::size_t>(kPos0 + kRows) * kKvHeads * kDim;
  std::vector<__half> h_k(kcn), h_v(kcn);
  std::uint32_t state = 12345U;
  auto rnd = [&] {
    state = state * 1664525U + 1013904223U;
    return (static_cast<float>(state >> 8) / 8388608.0F) - 1.0F;
  };
  for (auto& x : h_q) x = rnd();
  for (std::size_t i = 0; i < kcn; ++i) {
    h_k[i] = __float2half_rn(2.0F * rnd());
    h_v[i] = __float2half_rn(2.0F * rnd());
  }
  float* q = nullptr;
  float* sink = nullptr;
  __half *k = nullptr, *v = nullptr;
  float* out[4] = {nullptr, nullptr, nullptr, nullptr};
  CHECK(hipMalloc(&q, kQn * sizeof(float)));
  CHECK(hipMalloc(&sink, kHeadsN * sizeof(float)));
  CHECK(hipMalloc(&k, kcn * sizeof(__half)));
  CHECK(hipMalloc(&v, kcn * sizeof(__half)));
  for (auto& o : out) CHECK(hipMalloc(&o, kQn * sizeof(float)));
  CHECK(hipMemcpy(q, h_q.data(), kQn * sizeof(float),
                  hipMemcpyHostToDevice));
  CHECK(
      hipMemcpy(sink, h_sink.data(), kHeadsN * sizeof(float),
                hipMemcpyHostToDevice));
  CHECK(hipMemcpy(k, h_k.data(), kcn * sizeof(__half),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(v, h_v.data(), kcn * sizeof(__half),
                  hipMemcpyHostToDevice));

  const char* names[4] = {"<1,8>", "<2,4>", "<4,2>", "<8,1>"};
  int rc = 0;
  for (const char* label : {"windowed513", "full"}) {
    const std::uint32_t window = label[0] == 'w' ? 513U : 0U;
    std::fprintf(stderr, "=== %s ===\n", label);
    for (int c = 0; c < 4; ++c) {
      const fr::WmmaArgs a = MakeArgs(q, k, v, sink, out[c], window);
      LaunchCfg<1, 8>(a);  // unused template ref keeps all instantiated
      switch (c) {
        case 0: LaunchCfg<1, 8>(a); break;
        case 1: LaunchCfg<2, 4>(a); break;
        case 2: LaunchCfg<4, 2>(a); break;
        default: LaunchCfg<8, 1>(a); break;
      }
    }
    CHECK(hipDeviceSynchronize());
    std::vector<float> ref(kQn), got(kQn);
    CHECK(hipMemcpy(ref.data(), out[1], kQn * sizeof(float),
                    hipMemcpyDeviceToHost));
    for (int c = 0; c < 4; ++c) {
      std::size_t mismatch = 0;
      if (c != 1) {
        CHECK(hipMemcpy(got.data(), out[c], kQn * sizeof(float),
                        hipMemcpyDeviceToHost));
        for (std::size_t i = 0; i < kQn; ++i) {
          if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) {
            ++mismatch;
          }
        }
      }
      const fr::WmmaArgs a = MakeArgs(q, k, v, sink, out[c], window);
      const double ms = TimeCfg(c, a, 20);
      std::printf("%-12s cfg %-6s %8.3f ms  mismatches=%zu\n", label,
                  names[c], ms, mismatch);
      if (mismatch != 0) rc = 1;
    }
  }
  std::printf("%s\n", rc == 0 ? "BITWISE OK" : "BITWISE MISMATCH");
  return rc;
}
