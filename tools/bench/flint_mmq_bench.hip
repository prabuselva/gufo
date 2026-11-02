// A/B harness for the Flint mmq fork: times the shipped Q8_0 entry points on
// the measured Flint shapes and stores/compares a bitwise reference so any
// pipeline restructure (double-buffered LDS staging) must reproduce the
// reference output exactly.
//
//   clang++ --offload-arch=gfx1151 --driver-mode=g++ -L/opt/rocm/lib \
//     --hip-link -O3 -std=c++20 -x hip \
//     -I src/models/flint/kernels/rocm/mmq \
//     tools/bench/flint_mmq_bench.hip -x none \
//     build/release/src/models/flint/libgufo_flint_mmq.a \
//     -lhipblas -o /tmp/flint_mmq_bench
//
// Usage: /tmp/flint_mmq_bench [--save ref.bin] [--ref ref.bin] [reps]
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <flint_mmq.h>

#define CHECK(call)                                                     \
  do {                                                                  \
    const hipError_t err__ = (call);                                    \
    if (err__ != hipSuccess) {                                          \
      std::fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                   hipGetErrorString(err__), __FILE__, __LINE__);       \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

// flint_mmq entry points return an int status (0 == success).
#define RK(call)                                                        \
  do {                                                                  \
    const int rc__ = (call);                                            \
    if (rc__ != 0) {                                                    \
      std::fprintf(stderr, "mmq error %d at %s:%d\n", rc__, __FILE__,  \
                   __LINE__);                                           \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

namespace {

// Deterministic host RNG so weights/activations are identical across runs.
struct Xorshift {
  std::uint32_t s = 0x9e3779b9u;
  std::uint32_t Next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float Uniform() {  // [-1, 1)
    return static_cast<float>(Next() >> 8) / 8388608.0f - 1.0f;
  }
};

struct Shape {
  const char *name;
  bool moe;
  int M;           // weight rows (output dim) per expert / dense
  int K;           // reduction dim
  int n_tokens;    // moe routing rows; dense uses M for nothing, see below
  int n_experts;   // moe only
  int expert_used; // moe only
  int tokens;      // dense token rows
};

// Shapes from the pp2048/pp8192 profile: dense q projection, routed gate/up
// and down at a full 2048-token chunk (~48 rows/expert).
constexpr Shape kShapes[] = {
    {"dense_2048x4096x2048", false, 4096, 2048, 0, 0, 0, 2048},
    {"moe_gateup_1024x2048_e256", true, 1024, 2048, 2048, 256, 6, 0},
    {"moe_down_2048x1024_e256", true, 2048, 1024, 2048, 256, 6, 0},
};

std::size_t Q8_0Bytes(int rows, int k) {
  return static_cast<std::size_t>(rows) * (k / 32) * 34;  // d (fp16) + qs
}

void FillWeights(std::vector<std::uint8_t> &w, int rows, int k, Xorshift &rng) {
  w.resize(Q8_0Bytes(rows, k));
  const int n_blocks = rows * (k / 32);
  for (int b = 0; b < n_blocks; ++b) {
    unsigned short d_bits = static_cast<unsigned short>(rng.Next() & 0x3fff);
    std::memcpy(&w[static_cast<std::size_t>(b) * 34], &d_bits, 2);
    for (int q = 0; q < 32; ++q) w[b * 34 + 2 + q] = rng.Next() & 0xff;
  }
}

void FillIds(std::vector<std::int32_t> &ids, int n_tokens, int used,
             int n_experts, Xorshift &rng) {
  ids.resize(static_cast<std::size_t>(n_tokens) * used);
  for (int t = 0; t < n_tokens; ++t) {
    const int base = static_cast<int>(rng.Next() % n_experts);
    for (int i = 0; i < used; ++i) {
      ids[static_cast<std::size_t>(t) * used + i] = (base + i * 7) % n_experts;
    }
  }
}

float Median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char **argv) {
  const char *save_path = nullptr;
  const char *ref_path = nullptr;
  int reps = 20;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--save") && i + 1 < argc) {
      save_path = argv[++i];
    } else if (!std::strcmp(argv[i], "--ref") && i + 1 < argc) {
      ref_path = argv[++i];
    } else {
      reps = std::atoi(argv[i]);
    }
  }

  RK(flint_mmq_init(0));

  Xorshift rng;
  std::vector<float> all_out;  // concatenated device outputs for the ref
  for (const Shape &s : kShapes) {
    const int out_rows = s.moe ? s.n_tokens * s.expert_used : s.tokens;
    const std::size_t out_n =
        static_cast<std::size_t>(out_rows) *
        (s.moe ? s.M : s.M);  // both write [rows][M]

    std::vector<std::uint8_t> w;
    const int w_experts = s.moe ? s.n_experts : 1;
    FillWeights(w, w_experts * s.M, s.K, rng);

    std::vector<float> x(static_cast<std::size_t>(
                             s.moe ? s.n_tokens : s.tokens) *
                         s.K);
    for (auto &v : x) v = rng.Uniform() * 0.25f;

    std::vector<std::int32_t> ids;
    if (s.moe) FillIds(ids, s.n_tokens, s.expert_used, s.n_experts, rng);

    void *d_w = nullptr;
    float *d_x = nullptr;
    float *d_out = nullptr;
    std::int32_t *d_ids = nullptr;
    CHECK(hipMalloc(&d_w, w.size()));
    CHECK(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_x, x.size() * sizeof(float)));
    CHECK(hipMemcpy(d_x, x.data(), x.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_out, out_n * sizeof(float)));
    if (s.moe) {
      CHECK(hipMalloc(&d_ids, ids.size() * sizeof(std::int32_t)));
      CHECK(hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(std::int32_t),
                      hipMemcpyHostToDevice));
    }

    auto launch = [&]() {
      if (s.moe) {
        RK(flint_mmq_q8_0_moe_raw(d_w, d_x, d_ids, d_out, s.M, s.K,
                                     s.n_tokens, s.n_experts, s.expert_used, 0));
      } else {
        RK(flint_mmq_q8_0_dense(d_w, d_x, d_out, s.M, s.tokens, s.K, 0));
      }
    };

    launch();  // warmup (module load, scratch alloc)
    launch();
    CHECK(hipDeviceSynchronize());

    hipEvent_t ev0, ev1;
    CHECK(hipEventCreate(&ev0));
    CHECK(hipEventCreate(&ev1));
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
      CHECK(hipEventRecord(ev0, 0));
      launch();
      CHECK(hipEventRecord(ev1, 0));
      CHECK(hipEventSynchronize(ev1));
      float t;
      CHECK(hipEventElapsedTime(&t, ev0, ev1));
      ms.push_back(t);
    }
    const double flops =
        2.0 * static_cast<double>(out_rows) * s.M * s.K;
    const double t_med = Median(ms);
    std::printf("%-26s %8.3f ms  %7.2f TFLOP/s\n", s.name, t_med,
                flops / (t_med * 1e-3) / 1e12);

    std::vector<float> out(out_n);
    CHECK(hipMemcpy(out.data(), d_out, out_n * sizeof(float),
                    hipMemcpyDeviceToHost));
    all_out.insert(all_out.end(), out.begin(), out.end());

    CHECK(hipEventDestroy(ev0));
    CHECK(hipEventDestroy(ev1));
    CHECK(hipFree(d_w));
    CHECK(hipFree(d_x));
    CHECK(hipFree(d_out));
    if (d_ids) CHECK(hipFree(d_ids));
  }

  if (save_path) {
    FILE *f = std::fopen(save_path, "wb");
    if (!f) {
      std::fprintf(stderr, "cannot write %s\n", save_path);
      return 1;
    }
    std::fwrite(all_out.data(), sizeof(float), all_out.size(), f);
    std::fclose(f);
    std::printf("saved %zu floats -> %s\n", all_out.size(), save_path);
  }
  if (ref_path) {
    FILE *f = std::fopen(ref_path, "rb");
    if (!f) {
      std::fprintf(stderr, "cannot read %s\n", ref_path);
      return 1;
    }
    std::vector<float> ref(all_out.size());
    const std::size_t got =
        std::fread(ref.data(), sizeof(float), ref.size(), f);
    std::fclose(f);
    std::size_t mism = 0;
    if (got == ref.size()) {
      for (std::size_t i = 0; i < ref.size(); ++i) {
        mism += std::memcmp(&ref[i], &all_out[i], sizeof(float)) != 0;
      }
    } else {
      mism = static_cast<std::size_t>(-1);
    }
    std::printf("bitwise vs %s: %s (%zu mismatches)\n", ref_path,
                mism == 0 ? "IDENTICAL" : "DIFFERENT", mism);
    return mism == 0 ? 0 : 1;
  }
  return 0;
}
