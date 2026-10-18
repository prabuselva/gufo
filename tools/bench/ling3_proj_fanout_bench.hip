// Feasibility probe: multi-stream fan-out of the independent per-layer KDA
// projection GEMVs (ling3 lever 5.8). KdaLayer emits six GEMVs from the one
// normed input x into disjoint scratch (ssm_q/k/v/f/g -> 4096x2560, ssm_beta
// -> 32x2560) before the conv/delta chain consumes them; today all six run
// serially on the legacy default stream. This probe drives the REAL shipped
// Gemv launcher over a weight ring and measures per-"layer" wall time with
// the six projections spread over 1..4 lanes (main + non-blocking side
// streams, joined after the last projection where production ConvSilu reads
// them). Same bytes are never replayed through L2 across ring passes.
#include "src/models/ling3_0/kernels/rocm/gemv.hip.cpp"

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

using namespace gufo::models::ling3_0::rocm;

constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kInner = 4096;   // ssm_head_dim * num_heads
constexpr std::uint32_t kHeads = 32;     // ssm_beta rows
constexpr std::uint32_t kLayers = 16;    // weight ring length
constexpr int kReps = 8;                 // ring passes per timed round
constexpr int kProj = 6;                 // q,k,v,f,g,beta

std::size_t RowBytes(std::uint32_t cols, GemvType t) {
  switch (t) {
    case GemvType::kQ4_K: return (cols / 256U) * 144U;
    case GemvType::kQ6_K: return (cols / 256U) * 210U;
    case GemvType::kQ8_0: return (cols / 32U) * 34U;
    default: return 0U;
  }
}

// Six projections with their dominant GGUF quant types (k,v via Q6_K; q,f,g
// via Q4_K; beta a 32-row Q4_K head projection).
struct Proj {
  void* ring[kLayers];
  GemvType type;
  std::uint32_t rows;
  std::uint32_t cols;
};

struct Buffers {
  Proj q, k, v, f, g, beta;
  float* x;
  float* oq;
  float* ok;
  float* ov;
  float* of;
  float* og;
  float* obeta;
  hipStream_t lane[4];       // [0]=main, [1..3]=non-blocking sides
  hipEvent_t ev_lane[4];     // record the last GEMV of each used side lane
};

void AllocRing(Proj& p, GemvType type, std::uint32_t rows, std::uint32_t cols) {
  p.type = type;
  p.rows = rows;
  p.cols = cols;
  const std::size_t bytes =
      static_cast<std::size_t>(rows) * RowBytes(cols, type);
  for (std::uint32_t l = 0; l < kLayers; ++l) {
    CU(hipMalloc(&p.ring[l], bytes));
    CU(hipMemset(p.ring[l], 0x49 + static_cast<int>(l), bytes));
  }
}

float* AllocF(std::size_t n) {
  float* p = nullptr;
  CU(hipMalloc(&p, n * sizeof(float)));
  CU(hipMemset(p, 0, 0 + n * sizeof(float)));
  return p;
}

// Distribute the six projections over `lanes` streams (round-robin, the big
// q/k/v first) and join every used side lane before ConvSilu would consume
// the outputs. proj[i] -> lane[i % lanes].
void IssueLayer(const Buffers& b, std::uint32_t l, int lanes) {
  const Proj* proj[kProj] = {&b.q, &b.k, &b.v, &b.f, &b.g, &b.beta};
  float* out[kProj] = {b.oq, b.ok, b.ov, b.of, b.og, b.obeta};
  for (std::uint32_t i = 0; i < kProj; ++i) {
    const int lane = static_cast<int>(i) % lanes;
    Gemv(proj[i]->ring[l], proj[i]->type, proj[i]->rows, proj[i]->cols,
         RowBytes(proj[i]->cols, proj[i]->type), b.x, out[i], b.lane[lane]);
  }
  for (int lane = 1; lane < lanes; ++lane) {
    CU(hipEventRecord(b.ev_lane[lane], b.lane[lane]));
  }
}

void Join(const Buffers& b, int lanes) {
  for (int lane = 1; lane < lanes; ++lane) {
    CU(hipStreamWaitEvent(b.lane[0], b.ev_lane[lane], 0));
  }
}

double RunRound(const Buffers& b, int lanes, hipEvent_t ev0, hipEvent_t ev1) {
  CU(hipEventRecord(ev0, b.lane[0]));
  for (int rep = 0; rep < kReps; ++rep) {
    for (std::uint32_t l = 0; l < kLayers; ++l) {
      IssueLayer(b, l, lanes);
      Join(b, lanes);
    }
  }
  CU(hipEventRecord(ev1, b.lane[0]));
  CU(hipEventSynchronize(ev1));
  float ms = 0.0f;
  CU(hipEventElapsedTime(&ms, ev0, ev1));
  return static_cast<double>(ms) * 1000.0 / (kReps * kLayers);
}

}  // namespace

int main() {
  Buffers b{};
  AllocRing(b.q, GemvType::kQ4_K, kInner, kHidden);
  AllocRing(b.k, GemvType::kQ6_K, kInner, kHidden);
  AllocRing(b.v, GemvType::kQ6_K, kInner, kHidden);
  AllocRing(b.f, GemvType::kQ4_K, kInner, kHidden);
  AllocRing(b.g, GemvType::kQ4_K, kInner, kHidden);
  AllocRing(b.beta, GemvType::kQ4_K, kHeads, kHidden);

  b.x = AllocF(kHidden);
  b.oq = AllocF(std::max<std::size_t>(kInner, kHeads));
  b.ok = AllocF(kInner);
  b.ov = AllocF(kInner);
  b.of = AllocF(kInner);
  b.og = AllocF(kInner);
  b.obeta = AllocF(kHeads);

  for (int i = 0; i < 4; ++i) {
    CU(hipStreamCreateWithFlags(&b.lane[i],
                                i == 0 ? hipStreamNonBlocking
                                       : hipStreamNonBlocking));
    CU(hipEventCreateWithFlags(&b.ev_lane[i], hipEventDisableTiming));
  }
  hipEvent_t ev0, ev1;
  CU(hipEventCreate(&ev0));
  CU(hipEventCreate(&ev1));

  for (int lanes = 1; lanes <= 4; ++lanes) RunRound(b, lanes, ev0, ev1);

  double med[4] = {1e30, 1e30, 1e30, 1e30};
  for (int round = 0; round < 6; ++round) {
    for (int lanes = 1; lanes <= 4; ++lanes) {
      med[lanes - 1] = std::min(med[lanes - 1], RunRound(b, lanes, ev0, ev1));
    }
  }
  for (int lanes = 1; lanes <= 4; ++lanes) {
    std::printf("%d lane%s  %8.1f us/layer", lanes, lanes > 1 ? "s" : "",
                med[lanes - 1]);
    if (lanes > 1) {
      std::printf("  (%+.1f%% vs 1 lane)", (med[lanes - 1] / med[0] - 1) * 100);
    }
    std::printf("\n");
  }
  return 0;
}
