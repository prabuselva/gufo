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
    const double rel = r > 1e-6 ? e / r : e / 1e-6;
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
