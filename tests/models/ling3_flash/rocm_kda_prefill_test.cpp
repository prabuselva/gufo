// Chunked-prefill checks for the Ling 3.0 Flash KDA tier: synthetic projected
// q/k/v, gate, beta and conv streams drive the prefill pipeline (batched conv,
// history advance, batched head norms, transcendental prep, register-resident
// delta loop, batched output norm) and the results are compared against the
// proven single-token decode chain run token by token over the same chunk with
// the same starting conv histories and delta states.
//
// Every case exercises both loop kernels: the snapshot-free call selects the
// 128-wide row-split fast path, and the snapshot call selects the generic
// warp kernel, whose per-token states are compared against the states the
// decode chain leaves after each token. Conv history rows must be bit-exact
// (raw projected rows, no arithmetic); attention outputs, final states and
// snapshots match up to float reassociation.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/ling3_flash/kernels/rocm/kernels.hpp"
#include "tests/models/ling3_flash/hip_test.hpp"

namespace q = gufo::models::ling3_flash::rocm;
namespace t = gufo::tests::ling3_flash;

namespace {

constexpr std::uint32_t kHeads = 4U;
constexpr std::uint32_t kDim = 128U;  // The row-split fast path's head_dim.
constexpr std::uint32_t kKern = 4U;   // The model's ssm_conv_kernel.
constexpr float kEps = 1e-6F;
constexpr float kGateLb = -5.0F;  // The model's kda_gate_lower_bound.

bool Check(const char* name, double worst, double tolerance) {
  std::cout << name << " worst error " << worst << '\n';
  if (!(worst <= tolerance)) {
    std::cerr << name << " exceeded tolerance " << tolerance << '\n';
    return false;
  }
  return true;
}

bool CheckExact(const char* name, const std::vector<float>& reference,
                const std::vector<float>& candidate) {
  if (reference.size() != candidate.size() ||
      std::memcmp(reference.data(), candidate.data(),
                  reference.size() * sizeof(float)) != 0) {
    std::cerr << name << " differs bit-wise\n";
    return false;
  }
  std::cout << name << " bit-exact\n";
  return true;
}

std::string CaseName(const char* what, std::uint32_t tokens) {
  return std::string(what) + "[T=" + std::to_string(tokens) + "]";
}

bool RunCase(std::uint32_t tokens) {
  const std::uint32_t inner = kHeads * kDim;
  const std::uint32_t hist = (kKern - 1U) * inner;
  const std::uint32_t state_count = kHeads * kDim * kDim;
  const std::uint32_t snap_rows = tokens < 5U ? tokens : 5U;
  const std::size_t snap_count =
      static_cast<std::size_t>(snap_rows) * kHeads * kDim * kDim;
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(kDim));

  const std::vector<float> q_proj = t::MakeValues(tokens * inner, 7U, 0.9F);
  const std::vector<float> k_proj = t::MakeValues(tokens * inner, 11U, 0.9F);
  const std::vector<float> v_proj = t::MakeValues(tokens * inner, 13U, 0.9F);
  const std::vector<float> f = t::MakeValues(tokens * inner, 17U, 1.2F);
  const std::vector<float> beta = t::MakeValues(tokens * kHeads, 19U, 2.0F);
  const std::vector<float> a = t::MakeValues(kHeads, 23U, 0.8F);
  const std::vector<float> g = t::MakeValues(tokens * inner, 29U, 1.0F);
  const std::vector<float> norm_w = t::MakeValues(kDim, 31U, 0.5F, 0.7F);
  const std::vector<float> w_q = t::MakeValues(inner * kKern, 37U, 0.5F);
  const std::vector<float> w_k = t::MakeValues(inner * kKern, 41U, 0.5F);
  const std::vector<float> w_v = t::MakeValues(inner * kKern, 43U, 0.5F);
  const std::vector<float> hist_q = t::MakeValues(hist, 47U, 0.7F);
  const std::vector<float> hist_k = t::MakeValues(hist, 53U, 0.7F);
  const std::vector<float> hist_v = t::MakeValues(hist, 59U, 0.7F);
  const std::vector<float> state0 = t::MakeValues(state_count, 61U, 0.25F);

  t::HipBuffer<float> d_wq(inner * kKern), d_wk(inner * kKern),
      d_wv(inner * kKern), d_qp(tokens * inner), d_kp(tokens * inner),
      d_vp(tokens * inner), d_f(tokens * inner), d_beta(tokens * kHeads),
      d_a(kHeads), d_g(tokens * inner), d_norm_w(kDim);
  t::Upload(&d_wq, w_q);
  t::Upload(&d_wk, w_k);
  t::Upload(&d_wv, w_v);
  t::Upload(&d_qp, q_proj);
  t::Upload(&d_kp, k_proj);
  t::Upload(&d_vp, v_proj);
  t::Upload(&d_f, f);
  t::Upload(&d_beta, beta);
  t::Upload(&d_a, a);
  t::Upload(&d_g, g);
  t::Upload(&d_norm_w, norm_w);

  // ---- Oracle: the decode kernels, one token at a time. ----
  t::HipBuffer<float> d_ref_hq(hist), d_ref_hk(hist), d_ref_hv(hist),
      d_snap_ref(snap_count == 0 ? 1 : snap_count), d_ref_state(state_count),
      d_cq(tokens * inner), d_ck(tokens * inner), d_cv(tokens * inner),
      d_out(tokens * inner);
  t::Upload(&d_ref_hq, hist_q);
  t::Upload(&d_ref_hk, hist_k);
  t::Upload(&d_ref_hv, hist_v);
  t::Upload(&d_ref_state, state0);
  for (std::uint32_t tok = 0; tok < tokens; ++tok) {
    const std::size_t off = static_cast<std::size_t>(tok) * inner;
    q::ConvSilu(d_wq.get(), d_qp.get() + off, d_ref_hq.get(), d_cq.get() + off,
                inner, kKern, nullptr);
    q::ConvSilu(d_wk.get(), d_kp.get() + off, d_ref_hk.get(), d_ck.get() + off,
                inner, kKern, nullptr);
    q::ConvSilu(d_wv.get(), d_vp.get() + off, d_ref_hv.get(), d_cv.get() + off,
                inner, kKern, nullptr);
    q::KdaNormHeads(d_cq.get() + off, d_ck.get() + off, kHeads, kDim, kEps,
                    q_scale, nullptr);
    q::KdaDelta(d_cq.get() + off, d_ck.get() + off, d_cv.get() + off,
                d_f.get() + off, d_beta.get() + tok * kHeads, d_a.get(),
                d_norm_w.get(), d_g.get() + off, d_ref_state.get(),
                d_out.get() + off, kHeads, kDim, kEps, kGateLb, nullptr);
    if (tok < snap_rows) {
      t::CheckHip(hipMemcpy(d_snap_ref.get() +
                                static_cast<std::size_t>(tok) * state_count,
                            d_ref_state.get(), state_count * sizeof(float),
                            hipMemcpyDeviceToDevice),
                  "snapshot reference");
    }
    t::CheckHip(hipGetLastError(), "decode oracle launch");
  }
  t::CheckHip(hipDeviceSynchronize(), "decode oracle sync");
  const std::vector<float> out_ref = t::Download(&d_out, tokens * inner);
  const std::vector<float> state_ref = t::Download(&d_ref_state, state_count);
  const std::vector<float> ref_hist_q = t::Download(&d_ref_hq, hist);
  const std::vector<float> ref_hist_k = t::Download(&d_ref_hk, hist);
  const std::vector<float> ref_hist_v = t::Download(&d_ref_hv, hist);
  const std::vector<float> snaps_ref =
      snap_count == 0 ? std::vector<float>{}
                      : t::Download(&d_snap_ref, snap_count);

  // ---- Prefill tier: batched front-end + both delta loops. ----
  // The decode oracle advanced d_ref_h* in place, so the prefill starts from
  // fresh uploads of the chunk-start histories; its own history-advance output
  // must land bit-exactly on what the oracle rolled.
  t::HipBuffer<float> d_hq0(hist), d_hk0(hist), d_hv0(hist),
      d_pq(tokens * inner), d_pk(tokens * inner), d_pv(tokens * inner),
      d_nq(hist), d_nk(hist), d_nv(hist), d_decay(tokens * inner),
      d_beta_pre(tokens * kHeads), d_kq_pre(tokens * kHeads),
      d_pf_state(state_count), d_attn(tokens * inner),
      d_loop_state(state_count), d_attn2(tokens * inner), d_snap(snap_count);
  t::Upload(&d_hq0, hist_q);
  t::Upload(&d_hk0, hist_k);
  t::Upload(&d_hv0, hist_v);
  q::KdaConvSiluPrefill(d_wq.get(), d_qp.get(), d_hq0.get(), d_pq.get(), tokens,
                        inner, kKern, nullptr);
  q::KdaConvSiluPrefill(d_wk.get(), d_kp.get(), d_hk0.get(), d_pk.get(), tokens,
                        inner, kKern, nullptr);
  q::KdaConvSiluPrefill(d_wv.get(), d_vp.get(), d_hv0.get(), d_pv.get(), tokens,
                        inner, kKern, nullptr);
  q::KdaHistoryUpdate(d_qp.get(), d_hq0.get(), d_nq.get(), tokens, inner, kKern,
                      nullptr);
  q::KdaHistoryUpdate(d_kp.get(), d_hk0.get(), d_nk.get(), tokens, inner, kKern,
                      nullptr);
  q::KdaHistoryUpdate(d_vp.get(), d_hv0.get(), d_nv.get(), tokens, inner, kKern,
                      nullptr);
  q::KdaNormHeadsPrefill(d_pq.get(), d_pk.get(), tokens, kHeads, kDim, kEps,
                         q_scale, nullptr);
  q::KdaPrep(d_f.get(), d_a.get(), d_beta.get(), d_pq.get(), d_pk.get(),
             kGateLb, d_decay.get(), d_beta_pre.get(), d_kq_pre.get(), tokens,
             kHeads, kDim, nullptr);
  // Fast path (no snapshots) plus the batched gated output norm.
  t::Upload(&d_pf_state, state0);
  q::KdaDeltaLoop(d_pq.get(), d_pk.get(), d_pv.get(), d_decay.get(),
                  d_beta_pre.get(), d_kq_pre.get(), d_pf_state.get(),
                  d_attn.get(), tokens, kHeads, kDim, nullptr, 0, nullptr);
  q::KdaOutNormPrefill(d_attn.get(), d_g.get(), d_norm_w.get(), tokens, kHeads,
                       kDim, kEps, nullptr);
  // Generic warp kernel with per-token snapshots, from the same start state.
  t::Upload(&d_loop_state, state0);
  q::KdaDeltaLoop(d_pq.get(), d_pk.get(), d_pv.get(), d_decay.get(),
                  d_beta_pre.get(), d_kq_pre.get(), d_loop_state.get(),
                  d_attn2.get(), tokens, kHeads, kDim, d_snap.get(), snap_rows,
                  nullptr);
  t::CheckHip(hipGetLastError(), "prefill tier launch");
  t::CheckHip(hipDeviceSynchronize(), "prefill tier sync");

  std::vector<float> out_pf = t::Download(&d_attn, tokens * inner);
  std::vector<float> state_pf = t::Download(&d_pf_state, state_count);
  std::vector<float> state_loop = t::Download(&d_loop_state, state_count);
  std::vector<float> snap_pf = t::Download(&d_snap, snap_count);

  bool ok = true;
  ok &= Check(CaseName("out", tokens).c_str(),
              t::WorstRelative(out_ref, out_pf), 1e-3);
  ok &= Check(CaseName("state", tokens).c_str(),
              t::WorstRelative(state_ref, state_pf), 2e-3);
  ok &= Check(CaseName("state-loop", tokens).c_str(),
              t::WorstRelative(state_ref, state_loop), 2e-3);
  for (std::uint32_t row = 0; row < snap_rows; ++row) {
    const std::size_t s = static_cast<std::size_t>(row) * state_count;
    const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(s);
    const std::ptrdiff_t e = static_cast<std::ptrdiff_t>(s + state_count);
    ok &= Check(
        CaseName(("snapshot[" + std::to_string(row) + "]").c_str(), tokens)
            .c_str(),
        t::WorstRelative({snaps_ref.begin() + b, snaps_ref.begin() + e},
                         {snap_pf.begin() + b, snap_pf.begin() + e}),
        2e-3);
  }
  ok &= CheckExact(CaseName("history-q", tokens).c_str(), ref_hist_q,
                   t::Download(&d_nq, hist));
  ok &= CheckExact(CaseName("history-k", tokens).c_str(), ref_hist_k,
                   t::Download(&d_nk, hist));
  ok &= CheckExact(CaseName("history-v", tokens).c_str(), ref_hist_v,
                   t::Download(&d_nv, hist));
  return ok;
}

}  // namespace

int main() {
  bool ok = true;
  for (const std::uint32_t tokens : {1U, 2U, 5U, 17U, 64U}) {
    ok &= RunCase(tokens);
  }
  return ok ? 0 : 1;
}
