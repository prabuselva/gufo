// Feasibility probe: two-stream overlap of the independent MoE decode
// branches (ling3 lever: routed experts vs always-on shared expert).
//
// Drives the REAL shipped launchers by including the kernel TU, so the
// instruction mix, launch shapes and DRAM access patterns are production.
// Per decode MoE layer (m=1, 512 experts top-8):
//   routed : GemvGrouped Q4_K gate + up (768x2560/expert), Q6_K down
//            (2560x768/expert);
//   shared : Gemv Q8_0 gate/up/down (768x2560, 2560x768).
// The elementwise Swiglu/Add kernels are skipped (1.9-2.4 us each, they stay
// in either stream in production too). Weights live in a 32-layer ring so
// repeated passes never replay the same bytes through L2. Variants:
//   A serial    : production single-stream order.
//   B shared    : shared chain on a non-blocking side stream; main joins per
//                 layer where Add consumes it.
//   C gu+shared : additionally split routed gate || up, main joins per layer
//                 where down consumes both.
#include "src/models/ling3_flash/kernels/rocm/gemv.hip.cpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define CU(x)                                                        \
  do {                                                               \
    hipError_t e_ = (x);                                             \
    if (e_ != hipSuccess) {                                          \
      std::fprintf(stderr, "HIP %s @%d: %s\n", #x, __LINE__,         \
                   hipGetErrorString(e_));                           \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

namespace {

using namespace gufo::models::ling3_flash::rocm;

constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kFf = 768;      // expert_ff and shared_expert_ff
constexpr std::uint32_t kUsed = 8;      // active experts per token
constexpr std::uint32_t kLayers = 32;   // weight ring length
constexpr int kReps = 6;                // ring passes per timed round

std::size_t RowBytes(std::uint32_t cols, GemvType t) {
  switch (t) {
    case GemvType::kQ4_K: return (cols / 256U) * 144U;
    case GemvType::kQ6_K: return (cols / 256U) * 210U;
    case GemvType::kQ8_0: return (cols / 32U) * 34U;
    default: return 0U;
  }
}

struct Buffers {
  void* gate[kLayers];
  void* up[kLayers];
  void* down[kLayers];
  void* sh_gate[kLayers];
  void* sh_up[kLayers];
  void* sh_down[kLayers];
  std::int32_t* ids;
  float *x, *m_gate, *m_up, *m_down, *h_gate, *h_up, *h_inter, *h_down;
  hipStream_t main_s, sh_s, up_s;
  hipEvent_t ev_shared, ev_up;  // recorded on sh_s / up_s, joined on main_s
};

void AllocRing(void** p, GemvType type, std::uint32_t experts,
               std::uint32_t rows, std::uint32_t cols) {
  const std::size_t bytes = static_cast<std::size_t>(experts) * rows *
                            RowBytes(cols, type);
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    CU(hipMalloc(&p[l], bytes));
    CU(hipMemset(p[l], 0x49 + static_cast<int>(l), bytes));
  }
}

float* AllocF(std::size_t n) {
  float* p = nullptr;
  CU(hipMalloc(&p, n * sizeof(float)));
  CU(hipMemset(p, 0, n * sizeof(float)));
  return p;
}

// Join the shared branch where production Add consumes it, and (variant C)
// join the up branch before down (production down reads swiglu(gate, up)).
void Join(const Buffers& b, bool wait_up, bool wait_shared) {
  if (wait_up) {
    CU(hipStreamWaitEvent(b.main_s, b.ev_up, 0));
  }
  if (wait_shared) {
    CU(hipStreamWaitEvent(b.main_s, b.ev_shared, 0));
  }
}

void IssueLayer(const Buffers& b, std::uint32_t l, bool shared_branch,
                bool split_gu) {
  const std::size_t gu_stride =
      static_cast<std::size_t>(kFf) * RowBytes(kHidden, GemvType::kQ4_K);
  const std::size_t down_stride =
      static_cast<std::size_t>(kHidden) * RowBytes(kFf, GemvType::kQ6_K);
  hipStream_t s_up = split_gu ? b.up_s : b.main_s;
  // Producer record: fork order comes from the previous layer's join, which
  // already waited on this layer's predecessors reading x (read-only).
  GemvGrouped(b.gate[l], GemvType::kQ4_K, gu_stride, b.ids, kUsed, kUsed, kFf,
              kHidden, b.x, 0, b.m_gate, b.main_s);
  GemvGrouped(b.up[l], GemvType::kQ4_K, gu_stride, b.ids, kUsed, kUsed, kFf,
              kHidden, b.x, 0, b.m_up, s_up);
  if (split_gu) {
    CU(hipEventRecord(b.ev_up, b.up_s));
  }
  GemvGrouped(b.down[l], GemvType::kQ6_K, down_stride, b.ids, kUsed, kUsed,
              kHidden, kFf, b.m_gate, 0, b.m_down, b.main_s);
  hipStream_t s_sh = shared_branch ? b.sh_s : b.main_s;
  const std::size_t hg_rb = RowBytes(kHidden, GemvType::kQ8_0);
  const std::size_t hd_rb = RowBytes(kFf, GemvType::kQ8_0);
  Gemv(b.sh_gate[l], GemvType::kQ8_0, kFf, kHidden, hg_rb, b.x, b.h_gate,
       s_sh);
  Gemv(b.sh_up[l], GemvType::kQ8_0, kFf, kHidden, hg_rb, b.x, b.h_up, s_sh);
  Gemv(b.sh_down[l], GemvType::kQ8_0, kHidden, kFf, hd_rb, b.h_gate, b.h_down,
       s_sh);
  if (shared_branch) {
    CU(hipEventRecord(b.ev_shared, b.sh_s));
  }
}

double RunRound(const Buffers& b, bool shared_branch, bool split_gu,
                hipEvent_t ev0, hipEvent_t ev1) {
  CU(hipEventRecord(ev0, b.main_s));
  for (int rep = 0; rep < kReps; ++rep) {
    for (std::uint32_t l = 0; l < kLayers; ++l) {
      IssueLayer(b, l, shared_branch, split_gu);
      Join(b, split_gu, shared_branch);
    }
  }
  CU(hipEventRecord(ev1, b.main_s));
  CU(hipEventSynchronize(ev1));
  float ms = 0.0f;
  CU(hipEventElapsedTime(&ms, ev0, ev1));
  return static_cast<double>(ms) * 1000.0 / (kReps * kLayers);
}

}  // namespace

int main() {
  Buffers b{};
  AllocRing(b.gate, GemvType::kQ4_K, kUsed, kFf, kHidden);
  AllocRing(b.up, GemvType::kQ4_K, kUsed, kFf, kHidden);
  AllocRing(b.down, GemvType::kQ6_K, kUsed, kHidden, kFf);
  AllocRing(b.sh_gate, GemvType::kQ8_0, 1U, kFf, kHidden);
  AllocRing(b.sh_up, GemvType::kQ8_0, 1U, kFf, kHidden);
  AllocRing(b.sh_down, GemvType::kQ8_0, 1U, kHidden, kFf);

  const std::size_t big = std::max<std::size_t>(kUsed * kFf, kHidden);
  b.x = AllocF(kHidden);
  for (float** p : {&b.m_gate, &b.m_up, &b.m_down, &b.h_gate, &b.h_up,
                    &b.h_inter, &b.h_down}) {
    *p = AllocF(big);
  }
  std::int32_t ids_h[kUsed];
  for (std::uint32_t s = 0; s < kUsed; ++s) {
    ids_h[s] = static_cast<std::int32_t>(s);
  }
  CU(hipMalloc(&b.ids, kUsed * sizeof(std::int32_t)));
  CU(hipMemcpy(b.ids, ids_h, kUsed * sizeof(std::int32_t),
               hipMemcpyHostToDevice));
  CU(hipStreamCreateWithFlags(&b.main_s, hipStreamNonBlocking));
  CU(hipStreamCreateWithFlags(&b.sh_s, hipStreamNonBlocking));
  CU(hipStreamCreateWithFlags(&b.up_s, hipStreamNonBlocking));
  CU(hipEventCreateWithFlags(&b.ev_shared, hipEventDisableTiming));
  CU(hipEventCreateWithFlags(&b.ev_up, hipEventDisableTiming));
  hipEvent_t ev0, ev1;
  CU(hipEventCreate(&ev0));
  CU(hipEventCreate(&ev1));

  RunRound(b, false, false, ev0, ev1);  // warm-up

  constexpr int kVariants = 3;
  double med[kVariants] = {1e30, 1e30, 1e30};
  for (int round = 0; round < 5; ++round) {
    for (int v = 0; v < kVariants; ++v) {
      const double t = RunRound(b, v >= 1, v == 2, ev0, ev1);
      med[v] = std::min(med[v], t);
    }
  }
  static const char* kNames[kVariants] = {"A serial   ", "B shared   ",
                                          "C gu+shared"};
  for (int v = 0; v < kVariants; ++v) {
    std::printf("%s %8.1f us/layer", kNames[v], med[v]);
    if (v > 0) {
      std::printf("  (%+.1f%% vs A)", (med[v] / med[0] - 1.0) * 100.0);
    }
    std::printf("\n");
  }
  return 0;
}
