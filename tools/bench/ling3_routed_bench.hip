// Standalone A/B for the Ling3 routed expert GEMM (RoutedF16GEMMKernel).
//
// The workload mirrors one production prefill MoE layer: 1024 tokens
// (kPrefillChunkRows), 8 routed slots over 512 experts with a skewed
// histogram, the fused gate/up pair launch (m=768, k=2560) and the down
// launch (m=2560, k=768), both Q4_K with 64-row macro tiles. Outputs are
// checked against a double oracle that dequantizes the same Q4_K bytes the
// kernel reads; later candidates must additionally be bit-identical to the
// shipped kernel's outputs.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "src/models/ling3_0/kernels/rocm/routed_f16.hip.cpp"

namespace q = gufo::models::ling3_0::rocm;

namespace {

constexpr std::uint32_t kHid = 2560;    // gate k / down m
constexpr std::uint32_t kFf = 768;      // gate m / down k
constexpr std::uint32_t kExperts = 512;
constexpr std::uint32_t kSlotsPerToken = 8;
constexpr std::uint32_t kTileRows = 64;

void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, hipGetErrorString(err));
    std::exit(1);
  }
}

void Need(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "%s rejected\n", what);
    std::exit(1);
  }
}

template<typename T>
T* Dev(std::size_t n) {
  T* p = nullptr;
  Check(hipMalloc(&p, n * sizeof(T)), "hipMalloc");
  return p;
}

std::uint64_t SplitMix(std::uint64_t& s) {
  s += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

std::size_t RowBytes(std::size_t k) { return (k / 256) * 144; }

// Element e of a 256-element superblock (the kernel's own layout): K block
// p = e / 32 sits in the 32-byte nibble group g = p / 2 at qs bytes 32g..
// 32g+31 (qs starts after the 16-byte header), taking the low (p even) or
// high (p odd) nibbles; elements 0-15 / 16-31 of a block are group words
// 0-15 / 16-31. The header pair at index p supplies (sc, mn).
double Q4KValue(const std::uint8_t* sb, int e) {
  const int p = e / 32;
  const int byte = 16 + 32 * (p / 2) + ((e % 32) / 16) * 16 + (e % 16);
  const int nib = p & 1;
  double s = static_cast<double>((sb[byte] >> (4 * nib)) & 0xF);
  const int b = p;
  double sc = 0.0;
  double mn = 0.0;
  if (b < 4) {
    sc = sb[4 + b] & 0x3F;
    mn = sb[8 + b] & 0x3F;
  } else {
    sc = (sb[8 + b] & 0x0F) | ((sb[b] >> 6) << 4);
    mn = (sb[8 + b] >> 4) | ((sb[4 + b] >> 6) << 4);
  }
  __half d, dm;
  std::memcpy(&d, sb, 2);
  std::memcpy(&dm, sb + 2, 2);
  // Mirror the kernel: scale and offset are products rounded back to f16,
  // and CodesToHalves rounds q * scale - offset to f16 in one fused step.
  const __half scale = __float2half_rn(__half2float(d) * static_cast<float>(sc));
  const __half offset =
      __float2half_rn(__half2float(dm) * static_cast<float>(mn));
  const __half w = __float2half_rn(s * __half2float(scale) - __half2float(offset));
  return __half2float(w);
}

double DotRow(const std::uint8_t* w_row, const __half* x, std::size_t k) {
  double acc = 0.0;
  for (std::size_t i = 0; i < k; ++i)
    acc += __half2float(x[i]) * Q4KValue(w_row + (i / 256) * 144, i % 256);
  return acc;
}

struct Fixture {
  std::uint32_t tokens = 0;
  std::uint32_t slots = 0;
  std::uint32_t rows_total = 0;
  std::uint32_t n_tiles = 0;
  std::vector<std::int32_t> tiles;
  std::vector<std::int32_t> rows_token;
  std::vector<std::int32_t> rows_slot;
  std::vector<std::int32_t> rows_expert;
  std::vector<std::int32_t> bounds;
  std::vector<__half> xh_host;
  std::uint8_t *w_gate = nullptr, *w_up = nullptr, *w_down = nullptr;
  __half* xh = nullptr;
  __half* inter = nullptr;
  float* down_f32 = nullptr;
  std::vector<std::uint8_t> gate0, up0, down0;  // probe expert's weight rows
  std::int32_t probe_token = 0, probe_slot = 0, probe_begin = 0;
  std::uint32_t probe_expert = 0;
  std::int32_t* d_bounds = nullptr;
  std::int32_t* d_rows_token = nullptr;
  std::int32_t* d_rows_slot = nullptr;
  std::int32_t* d_tiles = nullptr;
};

void FillQ4K(std::vector<std::uint8_t>* w, std::size_t row_begin,
             std::size_t rows, std::size_t k, std::uint64_t& rng) {
  const std::size_t rb = RowBytes(k);
  for (std::size_t r = 0; r < rows; ++r) {
    std::uint8_t* row = w->data() + row_begin + r * rb;
    for (std::size_t sb = 0; sb < k / 256; ++sb) {
      std::uint8_t* p = row + sb * 144;
      const __half d =
          __float2half_rn(4.0e-4F * (static_cast<float>(SplitMix(rng) & 0xFFFF) /
                                         65536.0F +
                                     0.25F));
      const __half dm = __float2half_rn(4.0e-4F *
                                        static_cast<float>(SplitMix(rng) & 0xFFFF) /
                                        65536.0F);
      std::memcpy(p, &d, 2);
      std::memcpy(p + 2, &dm, 2);
      for (int i = 4; i < 144; ++i)
        p[i] = static_cast<std::uint8_t>(SplitMix(rng));
    }
  }
}

void BuildFixture(Fixture* f, double alpha, std::uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::gamma_distribution<double> g(alpha, 1.0);
  std::vector<double> weight(kExperts);
  double sum = 0.0;
  for (auto& v : weight) {
    v = g(gen) + 1e-6;
    sum += v;
  }
  f->slots = f->tokens * kSlotsPerToken;
  std::vector<std::int32_t> counts(kExperts);
  std::int64_t placed = 0;
  for (std::uint32_t e = 0; e + 1 < kExperts; ++e) {
    const std::int64_t c =
        std::llround(static_cast<double>(f->slots) * weight[e] / sum);
    counts[e] = static_cast<std::int32_t>(std::max<std::int64_t>(0, c));
    placed += counts[e];
  }
  counts[kExperts - 1] = std::max<std::int64_t>(0, f->slots - placed);
  placed += counts[kExperts - 1];
  while (placed > f->slots) {
    for (std::uint32_t e = kExperts; e-- > 0 && placed > f->slots;) {
      if (counts[e] > 0) {
        --counts[e];
        --placed;
      }
    }
  }

  f->bounds.assign(kExperts + 1, 0);
  std::int32_t run = 0;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    f->bounds[e] = run;
    for (std::uint32_t j = 0; j < (counts[e] + kTileRows - 1) / kTileRows;
         ++j) {
      f->tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
    }
    run += (counts[e] + 15) / 16 * 16;
  }
  f->bounds[kExperts] = run;
  f->rows_total = static_cast<std::uint32_t>(run);
  f->n_tiles = static_cast<std::uint32_t>(f->tiles.size());

  f->rows_token.assign(f->rows_total, -1);
  f->rows_slot.assign(f->rows_total, -1);
  f->rows_expert.assign(f->rows_total, -1);
  std::vector<std::int32_t> slot_ids(f->slots);
  for (std::uint32_t s = 0; s < f->slots; ++s)
    slot_ids[s] = static_cast<std::int32_t>(s);
  std::uint64_t sshuf = 555ULL;
  for (std::uint32_t s = f->slots; s > 1; --s)
    std::swap(slot_ids[s - 1], slot_ids[SplitMix(sshuf) % s]);
  std::int32_t next_slot = 0;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    const std::int32_t begin = f->bounds[e];
    for (std::int32_t c = 0; c < counts[e]; ++c) {
      const std::int32_t s = slot_ids[next_slot++];
      f->rows_token[begin + c] = s / static_cast<std::int32_t>(kSlotsPerToken);
      f->rows_slot[begin + c] = s;
      f->rows_expert[begin + c] = static_cast<std::int32_t>(e);
    }
  }

  f->xh_host.resize(static_cast<std::size_t>(f->tokens) * kHid);
  std::normal_distribution<float> n(0.0F, 1.0F);
  for (auto& v : f->xh_host) v = __float2half_rn(n(gen));

  const std::size_t gate_slab = kFf * RowBytes(kHid);
  const std::size_t down_slab = kHid * RowBytes(kFf);
  std::vector<std::uint8_t> w_gate(kExperts * gate_slab);
  std::vector<std::uint8_t> w_up(kExperts * gate_slab);
  std::vector<std::uint8_t> w_down(kExperts * down_slab);
  std::uint64_t wrng = 918273645ULL;
  for (std::uint32_t e = 0; e < kExperts; ++e) {
    FillQ4K(&w_gate, static_cast<std::size_t>(e) * gate_slab, kFf, kHid, wrng);
    FillQ4K(&w_up, static_cast<std::size_t>(e) * gate_slab, kFf, kHid, wrng);
    FillQ4K(&w_down, static_cast<std::size_t>(e) * down_slab, kHid, kFf, wrng);
  }

  std::uint32_t e0 = 0;
  while (e0 < kExperts && counts[e0] == 0) ++e0;
  f->probe_token = f->rows_token[f->bounds[e0]];
  f->probe_slot = f->rows_slot[f->bounds[e0]];
  f->probe_begin = static_cast<std::int32_t>(f->bounds[e0]);
  f->probe_expert = e0;
  f->gate0.assign(w_gate.begin() + e0 * gate_slab,
                  w_gate.begin() + (e0 + 1) * gate_slab);
  f->up0.assign(w_up.begin() + e0 * gate_slab, w_up.begin() + (e0 + 1) * gate_slab);
  f->down0.assign(w_down.begin() + e0 * down_slab,
                  w_down.begin() + (e0 + 1) * down_slab);

  f->w_gate = Dev<std::uint8_t>(w_gate.size());
  f->w_up = Dev<std::uint8_t>(w_up.size());
  f->w_down = Dev<std::uint8_t>(w_down.size());
  Check(hipMemcpy(f->w_gate, w_gate.data(), w_gate.size(),
                  hipMemcpyHostToDevice), "w_gate copy");
  Check(hipMemcpy(f->w_up, w_up.data(), w_up.size(), hipMemcpyHostToDevice),
        "w_up copy");
  Check(hipMemcpy(f->w_down, w_down.data(), w_down.size(),
                  hipMemcpyHostToDevice), "w_down copy");
  f->xh = Dev<__half>(f->xh_host.size());
  Check(hipMemcpy(f->xh, f->xh_host.data(), f->xh_host.size() * 2,
                  hipMemcpyHostToDevice), "xh copy");
  f->inter = Dev<__half>(static_cast<std::size_t>(f->slots) * kFf);
  f->down_f32 = Dev<float>(static_cast<std::size_t>(f->slots) * kHid);
  f->d_bounds = Dev<std::int32_t>(f->bounds.size());
  f->d_rows_token = Dev<std::int32_t>(f->rows_token.size());
  f->d_rows_slot = Dev<std::int32_t>(f->rows_slot.size());
  f->d_tiles = Dev<std::int32_t>(f->tiles.size());
  Check(hipMemcpy(f->d_bounds, f->bounds.data(),
                  f->bounds.size() * 4, hipMemcpyHostToDevice), "bounds copy");
  Check(hipMemcpy(f->d_rows_token, f->rows_token.data(),
                  f->rows_token.size() * 4, hipMemcpyHostToDevice),
        "rows_token copy");
  Check(hipMemcpy(f->d_rows_slot, f->rows_slot.data(),
                  f->rows_slot.size() * 4, hipMemcpyHostToDevice),
        "rows_slot copy");
  Check(hipMemcpy(f->d_tiles, f->tiles.data(), f->tiles.size() * 4,
                  hipMemcpyHostToDevice), "tiles copy");
}

// One MoE layer's routed launches, exactly the executor's shapes.
void RunLayer(const Fixture& f) {
  Need(q::RoutedGatedF16Gemm(f.w_gate, f.w_up, q::WeightType::kQ4_K, f.xh,
                             f.d_tiles, f.n_tiles, kTileRows, f.d_bounds,
                             f.d_rows_token, f.d_rows_slot, 0.0F, f.inter, kFf,
                             kHid, nullptr),
        "gate/up launch");
  Need(q::RoutedF16Gemm(f.w_down, q::WeightType::kQ4_K, f.inter, f.d_tiles,
                        f.n_tiles, kTileRows, f.d_bounds, f.d_rows_slot,
                        f.d_rows_slot, nullptr, 0.0F, f.down_f32, nullptr, kHid,
                        kFf, kHid, nullptr),
        "down launch");
}

double TimeLaunch(const Fixture& f, bool gate, int iters) {
  hipEvent_t a, b;
  hipEventCreate(&a);
  hipEventCreate(&b);
  std::vector<double> ms;
  for (int i = 0; i < iters; ++i) {
    hipEventRecord(a);
    if (gate) {
      (void)q::RoutedGatedF16Gemm(f.w_gate, f.w_up, q::WeightType::kQ4_K, f.xh,
                                  f.d_tiles, f.n_tiles, kTileRows, f.d_bounds,
                                  f.d_rows_token, f.d_rows_slot, 0.0F, f.inter,
                                  kFf, kHid, nullptr);
    } else {
      (void)q::RoutedF16Gemm(f.w_down, q::WeightType::kQ4_K, f.inter,
                             f.d_tiles, f.n_tiles, kTileRows, f.d_bounds,
                             f.d_rows_slot, f.d_rows_slot, nullptr, 0.0F,
                             f.down_f32, nullptr, kHid, kFf, kHid, nullptr);
    }
    hipEventRecord(b);
    hipEventSynchronize(b);
    float t = 0.0F;
    hipEventElapsedTime(&t, a, b);
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  hipEventDestroy(a);
  hipEventDestroy(b);
  return ms[ms.size() / 2] * 1000.0;  // µs
}

std::uint64_t Fnv(const void* p, std::size_t n, std::uint64_t h) {
  const std::uint8_t* b = static_cast<const std::uint8_t*>(p);
  for (std::size_t i = 0; i < n; ++i) {
    h = (h ^ b[i]) * 0x100000001B3ULL;
  }
  return h;
}

// Identity-decode probe: encode element e of every superblock of down row 0
// as the nibble (e % 16) at the oracle position (pass A) and (e / 16) at the
// oracle position (pass B), with all scales at 1, so the kernel's own
// one-hot output for k reports (e' % 16, e' / 16) for the element e' it
// decoded at k. The printed pairs reconstruct the kernel's true mapping.
std::uint64_t ProbePass(const Fixture& f, bool high) {
  std::vector<std::uint8_t> slab(static_cast<std::size_t>(kHid) *
                                     RowBytes(kFf),
                                 0);
  for (std::size_t sb = 0; sb < kFf / 256; ++sb) {
    std::uint8_t* h = slab.data() + sb * 144;  // row 0's superblock header
    const __half d = __float2half_rn(1.0F);
    std::memcpy(h, &d, 2);
    h[4] = h[5] = h[6] = h[7] = 0x01;
    h[8] = h[9] = h[10] = h[11] = 0x00;
    h[12] = h[13] = h[14] = h[15] = 0x01;
    for (int e = 0; e < 256; ++e) {
      const int v = high ? (e >> 4) : (e & 15);
      const int p = e / 32;
      const int byte = 16 + 32 * (p / 2) + (e % 32);
      h[byte] = static_cast<std::uint8_t>(p & 1 ? h[byte] | (v << 4) : v);
    }
  }
  Check(hipMemcpy(f.w_down, slab.data(), slab.size(), hipMemcpyHostToDevice),
        "probe slab upload");
  const __half one = __float2half_rn(1.0F);
  std::uint64_t key = 0;
  std::vector<float> got(1);
  for (std::size_t k = 0; k < kFf; ++k) {
    Check(hipMemset(f.inter, 0, static_cast<std::size_t>(f.slots) * kFf * 2),
          "probe memset");
    Check(hipMemcpy(f.inter + k, &one, 2, hipMemcpyHostToDevice), "one set");
    (void)q::RoutedF16Gemm(f.w_down, q::WeightType::kQ4_K, f.inter, f.d_tiles,
                           f.n_tiles, kTileRows, f.d_bounds, f.d_rows_slot,
                           f.d_rows_slot, nullptr, 0.0F, f.down_f32, nullptr,
                           kHid, kFf, kHid, nullptr);
    Check(hipDeviceSynchronize(), "probe sync");
    Check(hipMemcpy(got.data(), f.down_f32, 4, hipMemcpyDeviceToHost),
          "got copy");
    key = key * 1000003ULL + static_cast<std::uint64_t>(got[0]);
    std::printf("%zu %d\n", k, static_cast<int>(got[0]));
  }
  return key;
}

// Scale-pair probe: q=1 for every element, d=1.0 (bytes 0-1), dmin=2.0
// (bytes 2-3), and sc/mn chosen per pair j so out[k] = sc_j' * d - mn_j' *
// dmin reveals which pair j' the kernel applies at element k. Pass S uses
// sc_j=j+1, mn_j=0 (expect w=j+1; d read from bytes 2-3 => 2*(j+1)). Pass M
// uses sc_j=1, mn_j=j+1 (expect w=1-2*(j+1); swapped d/dmin => j-based
// small values around 1).
void ProbeScalePass(const Fixture& f, bool mn_mode) {
  std::vector<std::uint8_t> slab(static_cast<std::size_t>(kHid) *
                                     RowBytes(kFf),
                                 0);
  for (std::size_t sb = 0; sb < kFf / 256; ++sb) {
    std::uint8_t* h = slab.data() + sb * 144;
    const __half d = __float2half_rn(1.0F), dmin = __float2half_rn(2.0F);
    std::memcpy(h, &d, 2);
    std::memcpy(h + 2, &dmin, 2);
    for (int j = 0; j < 8; ++j) {
      const int S = mn_mode ? 1 : j + 1;
      const int M = mn_mode ? j + 1 : 0;
      if (j < 4) {
        h[4 + j] = static_cast<std::uint8_t>(S);
        h[8 + j] = static_cast<std::uint8_t>(M);
      } else {
        h[12 + (j - 4)] = static_cast<std::uint8_t>((M << 4) | S);
      }
    }
    for (int b = 16; b < 144; ++b) h[b] = 0x11;
  }
  Check(hipMemcpy(f.w_down, slab.data(), slab.size(), hipMemcpyHostToDevice),
        "scale slab upload");
  const __half one = __float2half_rn(1.0F);
  std::vector<float> got(1);
  for (std::size_t k = 0; k < kFf; ++k) {
    Check(hipMemset(f.inter, 0, static_cast<std::size_t>(f.slots) * kFf * 2),
          "scale memset");
    Check(hipMemcpy(f.inter + k, &one, 2, hipMemcpyHostToDevice), "one set");
    (void)q::RoutedF16Gemm(f.w_down, q::WeightType::kQ4_K, f.inter, f.d_tiles,
                           f.n_tiles, kTileRows, f.d_bounds, f.d_rows_slot,
                           f.d_rows_slot, nullptr, 0.0F, f.down_f32, nullptr,
                           kHid, kFf, kHid, nullptr);
    Check(hipDeviceSynchronize(), "scale sync");
    Check(hipMemcpy(got.data(), f.down_f32, 4, hipMemcpyDeviceToHost),
          "scale copy");
    std::printf("%zu %f\n", k, got[0]);
  }
}

// One-hot every element of the real random weights through the production
// launches: with x hot at (token, k), inter[slot][r] = gate_w[e][r][k] passed
// through the epilogue, and down_out[slot][r] = down_w[e][r][k] one by one.
// A full dense row through both launches then validates accumulation across
// the whole k range. This validates the kernel's reading of actual FillQ4K
// bytes end to end.
int ProbeRealBytes(const Fixture& f) {
  const auto t = static_cast<std::size_t>(f.probe_token);
  const auto slot = static_cast<std::size_t>(f.probe_slot);
  const __half one = __float2half_rn(1.0F);
  std::vector<__half> inter(static_cast<std::size_t>(kFf));
  std::vector<float> out(static_cast<std::size_t>(kHid));
  int gmis = 0, dmis = 0;
  auto close = [](double got, double want) {
    return std::abs(got - want) <= 2e-4 + 2e-2 * std::abs(want);
  };
  for (std::size_t k = 0; k < kHid; ++k) {
    Check(hipMemset(f.xh, 0, static_cast<std::size_t>(f.tokens) * kHid * 2),
          "real gate memset");
    Check(hipMemcpy(f.xh + t * kHid + k, &one, 2, hipMemcpyHostToDevice),
          "real gate hot");
    (void)q::RoutedGatedF16Gemm(f.w_gate, f.w_up, q::WeightType::kQ4_K, f.xh,
                                f.d_tiles, f.n_tiles, kTileRows, f.d_bounds,
                                f.d_rows_token, f.d_rows_slot, 0.0F, f.inter,
                                kFf, kHid, nullptr);
    Check(hipDeviceSynchronize(), "real gate sync");
    Check(hipMemcpy(inter.data(), f.inter + slot * kFf, kFf * 2,
                    hipMemcpyDeviceToHost),
          "real gate read");
    for (std::size_t r = 0; r < kFf; ++r) {
      const double gg = Q4KValue(f.gate0.data() + r * RowBytes(kHid) +
                                     (k / 256) * 144, k % 256);
      const double uu = Q4KValue(f.up0.data() + r * RowBytes(kHid) +
                                     (k / 256) * 144, k % 256);
      const double want = uu * (gg / (1.0 + std::exp(-gg)));
      const double got = __half2float(inter[r]);
      if (!close(got, want)) {
        if (gmis < 6)
          std::printf("GMIS k=%zu r=%zu got=%f want=%f\n", k, r, got, want);
        ++gmis;
      }
    }
  }
  for (std::size_t k = 0; k < kFf; ++k) {
    Check(hipMemset(f.inter, 0, static_cast<std::size_t>(f.slots) * kFf * 2),
          "real down memset");
    Check(hipMemcpy(f.inter + slot * kFf + k, &one, 2, hipMemcpyHostToDevice),
          "real down hot");
    (void)q::RoutedF16Gemm(f.w_down, q::WeightType::kQ4_K, f.inter, f.d_tiles,
                           f.n_tiles, kTileRows, f.d_bounds, f.d_rows_slot,
                           f.d_rows_slot, nullptr, 0.0F, f.down_f32, nullptr,
                           kHid, kFf, kHid, nullptr);
    Check(hipDeviceSynchronize(), "real down sync");
    Check(hipMemcpy(out.data(), f.down_f32 + slot * kHid, kHid * 4,
                    hipMemcpyDeviceToHost),
          "real down read");
    for (std::size_t r = 0; r < kHid; ++r) {
      const double want = Q4KValue(f.down0.data() + r * RowBytes(kFf) +
                                       (k / 256) * 144, k % 256);
      if (!close(out[r], want)) {
        if (dmis < 6)
          std::printf("DMIS k=%zu r=%zu got=%f want=%f\n", k, r,
                      static_cast<double>(out[r]), want);
        ++dmis;
      }
    }
  }
  {  // Dense rows through both launches validate full-k accumulation.
    std::mt19937 dgen(4242);
    std::normal_distribution<float> dn(0.0F, 1.0F);
    std::vector<__half> xh_dense(static_cast<std::size_t>(f.tokens) * kHid);
    for (auto& v : xh_dense) v = __float2half_rn(dn(dgen));
    Check(hipMemcpy(f.xh, xh_dense.data(), xh_dense.size() * 2,
                    hipMemcpyHostToDevice),
          "dense gate x");
    (void)q::RoutedGatedF16Gemm(f.w_gate, f.w_up, q::WeightType::kQ4_K, f.xh,
                                f.d_tiles, f.n_tiles, kTileRows, f.d_bounds,
                                f.d_rows_token, f.d_rows_slot, 0.0F, f.inter,
                                kFf, kHid, nullptr);
    Check(hipDeviceSynchronize(), "dense gate sync");
    Check(hipMemcpy(inter.data(), f.inter + slot * kFf, kFf * 2,
                    hipMemcpyDeviceToHost),
          "dense gate read");
    const __half* xt = xh_dense.data() + t * kHid;
    for (std::size_t r = 0; r < kFf; ++r) {
      const double gg =
          DotRow(f.gate0.data() + r * RowBytes(kHid), xt, kHid);
      const double uu =
          DotRow(f.up0.data() + r * RowBytes(kHid), xt, kHid);
      const double want = uu * (gg / (1.0 + std::exp(-gg)));
      const double got = __half2float(inter[r]);
      if (!close(got, want)) {
        if (gmis < 6)
          std::printf("GMIS-D got=%f want=%f\n", got, want);
        ++gmis;
      }
    }
    std::vector<__half> inter_dense(static_cast<std::size_t>(f.slots) * kFf);
    for (auto& v : inter_dense) v = __float2half_rn(dn(dgen));
    Check(hipMemcpy(f.inter, inter_dense.data(), inter_dense.size() * 2,
                    hipMemcpyHostToDevice),
          "dense down x");
    (void)q::RoutedF16Gemm(f.w_down, q::WeightType::kQ4_K, f.inter, f.d_tiles,
                           f.n_tiles, kTileRows, f.d_bounds, f.d_rows_slot,
                           f.d_rows_slot, nullptr, 0.0F, f.down_f32, nullptr,
                           kHid, kFf, kHid, nullptr);
    Check(hipDeviceSynchronize(), "dense down sync");
    Check(hipMemcpy(out.data(), f.down_f32 + slot * kHid, kHid * 4,
                    hipMemcpyDeviceToHost),
          "dense down read");
    const __half* xs = inter_dense.data() + slot * kFf;
    for (std::size_t r = 0; r < kHid; ++r) {
      const double want = DotRow(f.down0.data() + r * RowBytes(kFf), xs, kFf);
      if (!close(out[r], want)) {
        if (dmis < 6)
          std::printf("DMIS-D r=%zu got=%f want=%f\n", r,
                      static_cast<double>(out[r]), want);
        ++dmis;
      }
    }
  }
  std::printf("REAL gate_mismatches=%d down_mismatches=%d token=%d slot=%d\n",
              gmis, dmis, f.probe_token, f.probe_slot);
  return gmis + dmis;
}

int Probe(Fixture& f) {
  std::printf("PASS R\n");
  ProbeRealBytes(f);
  // Arm the single-tile identity fixture for PASS A/B/S/M only; PASS R runs
  // on the pristine bucket bounds/tiles/rows_slot.
  {
    const std::vector<std::int32_t> t1 = {0};
    std::vector<std::int32_t> bo(kExperts + 1, 16);
    std::vector<std::int32_t> rs(f.rows_total, -1);
    bo[0] = 0;
    rs[0] = 0;
    Check(hipMemcpy(f.d_tiles, t1.data(), 4, hipMemcpyHostToDevice), "p tiles");
    Check(hipMemcpy(f.d_bounds, bo.data(), bo.size() * 4,
                    hipMemcpyHostToDevice),
          "p bounds");
    Check(hipMemcpy(f.d_rows_slot, rs.data(), rs.size() * 4,
                    hipMemcpyHostToDevice),
          "p slot");
    f.n_tiles = 1;
  }
  std::printf("PASS A\n");
  ProbePass(f, false);
  std::printf("PASS B\n");
  ProbePass(f, true);
  std::printf("PASS S\n");
  ProbeScalePass(f, false);
  std::printf("PASS M\n");
  ProbeScalePass(f, true);
  return 0;
}

int Run(int argc, char** argv) {
  std::uint32_t tokens = 1024;
  double alpha = 0.09;
  int iters = 7;
  bool probe = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--tokens") tokens = std::atoi(argv[++i]);
    else if (a == "--alpha") alpha = std::atof(argv[++i]);
    else if (a == "--iters") iters = std::atoi(argv[++i]);
    else if (a == "--probe") probe = true;
  }
  Fixture f;
  f.tokens = tokens;
  BuildFixture(&f, alpha, 12345);
  std::fprintf(stderr, "tokens=%u slots=%u rows=%u tiles=%u\n", f.tokens,
               f.slots, f.rows_total, f.n_tiles);
  if (probe) return Probe(f);

  Check(hipMemset(f.inter, 0, static_cast<std::size_t>(f.slots) * kFf * 2),
        "inter memset");
  Check(hipMemset(f.down_f32, 0, static_cast<std::size_t>(f.slots) * kHid * 4),
        "down memset");
  RunLayer(f);
  Check(hipDeviceSynchronize(), "sync");
  Need(hipGetLastError() == hipSuccess, "kernel error");
  std::vector<__half> inter_host(static_cast<std::size_t>(f.slots) * kFf);
  std::vector<float> down_host(static_cast<std::size_t>(f.slots) * kHid);
  Check(hipMemcpy(inter_host.data(), f.inter, inter_host.size() * 2,
                  hipMemcpyDeviceToHost), "inter copy");
  Check(hipMemcpy(down_host.data(), f.down_f32, down_host.size() * 4,
                  hipMemcpyDeviceToHost), "down copy");
  std::size_t nz_i = 0, nz_d = 0;
  for (const __half v : inter_host)
    nz_i += __builtin_bit_cast(std::uint16_t, v) != 0;
  for (const float v : down_host) nz_d += v != 0.0F;
  std::fprintf(stderr, "debug nonzero inter=%zu/%zu down=%zu/%zu\n", nz_i,
               inter_host.size(), nz_d, down_host.size());

  // Oracle: sampled compact rows against double-precision dequantized GEMM.
  std::vector<std::uint8_t> gate_w(kFf * RowBytes(kHid));
  std::vector<std::uint8_t> up_w(kFf * RowBytes(kHid));
  std::vector<std::uint8_t> down_w(kHid * RowBytes(kFf));
  const std::size_t gate_slab = kFf * RowBytes(kHid);
  const std::size_t down_slab = kHid * RowBytes(kFf);
  std::uint64_t s = 99;
  int cached_e = -1, checked_rows = 0;
  double worst_gate = 0.0, worst_down = 0.0;
  int nans = 0;
  for (int rep = 0; rep < 24; ++rep) {
    const int c = static_cast<int>(SplitMix(s) % f.rows_total);
    const std::int32_t t = f.rows_token[c];
    const std::int32_t e = f.rows_expert[c];
    if (t < 0 || e < 0) continue;
    if (e != cached_e) {
      Check(hipMemcpy(gate_w.data(),
                      f.w_gate + static_cast<std::size_t>(e) * gate_slab,
                      gate_w.size(), hipMemcpyDeviceToHost), "gate slab");
      Check(hipMemcpy(up_w.data(), f.w_up + static_cast<std::size_t>(e) * gate_slab,
                      up_w.size(), hipMemcpyDeviceToHost), "up slab");
      Check(hipMemcpy(down_w.data(),
                      f.w_down + static_cast<std::size_t>(e) * down_slab,
                      down_w.size(), hipMemcpyDeviceToHost), "down slab");
      cached_e = e;
    }
    ++checked_rows;
    const std::size_t slot = static_cast<std::size_t>(f.rows_slot[c]);
    const __half* xr = f.xh_host.data() + static_cast<std::size_t>(t) * kHid;
    for (std::size_t r = 0; r < kFf; ++r) {
      const std::uint8_t* gr = gate_w.data() + r * RowBytes(kHid);
      const std::uint8_t* ur = up_w.data() + r * RowBytes(kHid);
      double gg = 0.0, uu = 0.0;
      for (std::size_t k = 0; k < kHid; ++k) {
        const double xv = __half2float(xr[k]);
        gg += xv * Q4KValue(gr + (k / 256) * 144, k % 256);
        uu += xv * Q4KValue(ur + (k / 256) * 144, k % 256);
      }
      const double want = uu * (gg / (1.0 + std::exp(-gg)));
      const double got = __half2float(inter_host[slot * kFf + r]);
      if (std::isnan(got)) ++nans;
      const double tol = 1e-3 + 5e-3 * std::abs(want);
      worst_gate = std::max(worst_gate, std::abs(got - want) / tol);
    }
    for (std::size_t rr = 0; rr < kHid; ++rr) {
      const std::uint8_t* dr = down_w.data() + rr * RowBytes(kFf);
      double dd = 0.0;
      for (std::size_t k = 0; k < kFf; ++k) {
        dd += __half2float(inter_host[slot * kFf + k]) *
              Q4KValue(dr + (k / 256) * 144, k % 256);
      }
      const double got = down_host[slot * kHid + rr];
      if (std::isnan(got)) ++nans;
      const double tol = 1e-3 + 5e-3 * std::abs(dd);
      worst_down = std::max(worst_down, std::abs(got - dd) / tol);
    }
  }

  const std::uint64_t sum_inter = Fnv(inter_host.data(), inter_host.size() * 2,
                                      0xcbf29ce484222325ULL);
  const std::uint64_t sum_down = Fnv(down_host.data(), down_host.size() * 4,
                                     0xcbf29ce484222325ULL);
  const double gate_us = TimeLaunch(f, true, iters);
  const double down_us = TimeLaunch(f, false, iters);
  std::printf(
      "baseline tiles=%u rows=%u gate=%8.1fus down=%8.1fus layer=%8.1fus "
      "pp1k40L=%7.2fms oracle gate=%.4f down=%.4f (%d rows) nans=%d "
      "sum=%016llx/%016llx\n",
      f.n_tiles, f.rows_total, gate_us, down_us, gate_us + down_us,
      (gate_us + down_us) * 40 / 1000.0, worst_gate, worst_down, checked_rows,
      nans, static_cast<unsigned long long>(sum_inter),
      static_cast<unsigned long long>(sum_down));
  return worst_gate <= 1.0 && worst_down <= 1.0 && nans == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) { return Run(argc, argv); }
