# IQ3_XXS Support Report

Status: **investigation complete, no code changed.** This report documents why
`run_rocm.sh` fails to load the Qwen3.8-Flash-Next UD-Q3_K_XL model and lays
out the full touch-list for adding IQ3_XXS support, for validation and use in
the next session.

## 1. The failure

```text
Error loading model '.../Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf':
Failed to open GGUF: Invalid GGUF tensor shape or unsupported storage type: blk.0.ffn_gate_exps.weight
```

- Raised at `src/core/gguf_reader.cpp:721-726`: `EncodedSizeBytes()` returns 0
  for an unknown `GgmlType`, which trips the shape/storage check.
- `GgmlType` (`src/core/gguf_reader.hpp:22-43`) jumps from `kIQ2_XXS = 16`
  (line 37) to `kIQ4_NL = 20` (line 38). **GGML type id 18 = IQ3_XXS is
  missing.**
- Not a build/sharding problem: the new llama.cpp split layout (shard 1 =
  metadata only, `n_tensors=0`, infos distributed over shards 2-3,
  `split.tensors.count=1224`) parses fine. Architecture `qwen4exp` is
  supported (`src/models/qwen38_flash_next/config.cpp:13`).

### Model file facts (verified by parsing all three shards)

- Path: `/home/praburaja/projects/llm/models/gguf/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q3_K_XL-0000{1,2,3}-of-00003.gguf`
- 1224 tensors total (shard 1: 0, shard 2: 402, shard 3: 822), quantized by
  Unsloth (`general.quantized_by = Unsloth`, imatrix).
- Tensor type histogram: `F32 ×557, Q8_0 ×502, IQ4_NL ×44, BF16 ×24, IQ3_XXS ×94, IQ4_XS ×2, Q6_K ×1`.
- **Only unsupported type: `IQ3_XXS` (94 tensors)** = the MoE expert weights
  `blk.N.ffn_gate_exps.weight` / `blk.N.ffn_up_exps.weight`
  (ndims=3, e.g. `[2560, 640, 512]`, type id 18). `ffn_down_exps` are IQ4_NL
  (id 20, already supported).

## 2. IQ3_XXS format (verified against the vendored copy in-repo)

Authoritative in-repo references (no external llama.cpp checkout needed):

- Block: `block_iq3_xxs` at
  `src/models/qwen38_flash_next/kernels/rocm/mmq/ggml-common.h:333-338` —
  `{ ggml_half d; uint8_t qs[3*QK_K/8]; }`, **98 bytes** per 256 elements
  (3.0625 bpw). The `static_assert` pins `2 + 3*(QK_K/8)`.
- Codebook: `iq3xxs_grid` (256 × `uint32_t`, each entry packs four 4-bit
  nibble values) at `.../mmq/ggml-common.h:913`.
- Device decode reference: `vec_dot_iq3_xxs_q8_1` at
  `src/models/qwen38_flash_next/kernels/rocm/mmq/vecdotq.hpp:1064-1095`.
  Per 32-element sub-block (8 sub-blocks per 256 block): 8 grid bytes →
  `iq3xxs_grid[]` nibbles; packed sign bits applied per element
  (`unpack_ksigns`); a 4-bit sub-block scale (`aux32 >> 28`) weights the
  sub-block dot product; final scale is `d × q8_1.ds`.

Do not re-derive the bit packing by hand — transcribe decode from
`vecdotq.hpp:1064` (device) and llama.cpp `dequantize_iq3_xxs` (CPU scalar).

## 3. Current IQ support tiers in gufo

"Supported" is tiered; IQ2_XXS is the precedent for enum+size only:

| Type | enum | size math | CPU dequant | CPU dot | qwen HIP kernels | qwen38 mmq |
|---|---|---|---|---|---|---|
| IQ2_XXS (16) | yes | 66 B/256 (`ggml_dequant.cpp:189-190`) | no | no | quantized-only | no |
| IQ4_NL (20) | yes | 18 B/32 | yes | yes | yes | no |
| IQ3_S (21) | yes | 110 B/256 | yes | yes | yes | no |
| IQ4_XS (23) | yes | 136 B/256 | yes | yes | yes | no |
| IQ3_XXS (18) | **no** | — | — | — | — | **device code vendored, not instantiated** |

Key insight: the vendored mmq layer already contains working IQ3_XXS device
code (`mmq.hpp:95,198,244,2845,3356-3357,4309`; `vecdotq.hpp:1061-1064`;
`common.hpp:767`) — it is simply never instantiated for gufo's MoE path.

## 4. Implementation touch-list (ordered)

### Phase 1 — core type + size math (unblocks GGUF loading)

1. `src/core/gguf_reader.hpp`
   - Enum: insert `kIQ3_XXS = 18` between lines 37/38.
   - `ToString` (45-89): add `case GgmlType::kIQ3_XXS: return "IQ3_XXS";`
     (match existing `"IQ4_NL"`-style casing).
2. `src/core/quant/ggml_dequant.hpp`
   - `block_iq3_xxs` struct (98 B) + `static_assert`, next to `block_iq3_s`
     (84-90); `QuantizedBlockElements` (109-133): `kIQ3_XXS → 256`.
   - Declare `DequantizeIQ3_XXS`, `DotProductIQ3_XXS`, `Iq3xxsGrid()`
     (mirror 180-200).
3. `src/core/quant/ggml_dequant.cpp`
   - `QuantizedRowBytes` (164-230): `block_bytes = sizeof(block_iq3_xxs)`
     beside the IQ cases at 210-218.
   - `kIq3xxsGrid[256]` — copy from `mmq/ggml-common.h:913` (follow the
     `kIq3sGrid[512]` pattern at 436-540; keep a single source of truth if a
     shared header is preferred).
   - `Iq3xxsValue` + `DequantizeIQ3_XXS` + `DotProductIQ3_XXS`
     (pattern: `Iq3sValue` 566-596, 624-674).

### Phase 2 — qwen38_flash_next loading (unblocks `run_rocm.sh` on CPU)

4. `src/models/qwen38_flash_next/weights.cpp`
   - `FormatOf` (19-41): add `kIQ3_XXS → {256, 98}`.
   - Expert whitelist line 143 (`{kQ4_K, kQ5_K, kQ6_K, ...}`): add
     `kIQ3_XXS` (gate/up experts; `ffn_down_exps` stays IQ4_NL via the
     projection/embedding role lists).
5. `src/models/qwen38_flash_next/cpu_ops.cpp`
   - `DequantizeRow` switch (44-84): add `case kIQ3_XXS`. **Trap: the default
     branch memsets zeros (line 82)** — without this case, experts silently
     decode to zero instead of erroring.

### Phase 3 — qwen reference model (shared GEMM route)

6. `src/models/qwen/weights.cpp` `SupportsTensorType` (59-90): add
   `kIQ3_XXS` to the projection whitelist (lines 82-87, beside
   `kIQ4_NL/kIQ4_XS/kIQ3_S`).
7. `src/models/qwen/gemm_route.hpp` `DescribeQwenGemmFormat` (58-100):
   decide tier — quantized-only (like IQ2_XXS, lines 95-97) or
   `cpu_direct`/`hip_direct` (like IQ4_NL/IQ3_S, 79-89). Suggest starting
   quantized-only until the HIP path is proven.
8. `src/models/qwen/hip/quant_ops.hpp`: `IQ3XXSBlock` (134-141 pattern),
   `QuantBlockBytes` → 98 (150-173), `QuantBlockQK` → 256 (175-198),
   `kDeviceIq3xxsGrid` (299 pattern), `DecodeQuantSub16` case
   (608-665 pattern), `IsSub16DecodedQuant` (695-700).
9. `src/models/qwen/hip/ops/gemm.hpp` `IsNativeWmmaQuant` (155-168) — only if
   WMMA-enabled; kernel switch sites listed in §5.

### Phase 4 — enable the vendored mmq path for MoE experts (HIP)

10. `src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp`
    - `qfn_mmq_moe_vec` type guard (359-363): add `GGML_TYPE_IQ3_XXS`.
    - Explicit instantiations (659-666): add
      `template void mul_mat_q_case<GGML_TYPE_IQ3_XXS>(...);` — the device
      kernels (`DECL_MMQ_CASE` at `mmq.hpp:4309`, traits at 3356) already
      exist.
11. `src/models/qwen38_flash_next/kernels/rocm/executor.cpp`
    - `Experts` tiled-path switch (991-1006) and `GatedExperts` gate
      (1021-1023): extend or document fallback to the vector path.
    - `MoeExperts` WMMA gate (1587-1593): IQ3_XXS experts must route to the
      mmq vector path (wmma requires Q4_K/Q5_K gate/up, Q5_1/Q8_0 down).
    - `DeviceModel::Upload` (273-288) only rejects Q6_K — IQ3_XXS raw bytes
      upload unchanged (weights go to device as-quantized;
      `device_model.cpp:46-79`).

### Phase 5 — tests

12. `tests/models/qwen/gemm_route_test.cpp` — add `kIQ3_XXS` to the type
    lists (43-44, 86-87), geometry check (`block_elements == 256`, pattern at
    98-101), route checks (284-292 pattern). Target `qwen_gemm_route_test`,
    labels `cpu;models;qwen;gemm;dispatch`.
13. `tests/core/ggml_dequant_test.cpp` — no IQ coverage exists today; add an
    IQ3_XXS round-trip (dequant vs scalar encode) + dot-product section.
    Labels `cpu;core;quant` (root `CMakeLists.txt:664-668`).
14. `tests/models/qwen/hip/quant/q4kxl_quant_ops_test.cpp` — CPU/GPU parity
    for `DequantizeIQ3_XXS`/`DotProductIQ3_XXS` (pattern 127-131, 153-159).
    Labels `hip;gpu;...;iqant` (tests/models/qwen/CMakeLists.txt:113-115).
15. Optional: `tests/models/qwen38_flash_next/ngram_test.cpp` (IQ4_NL fixture
    pattern at 216-263) — only if `ple_table` ever accepts IQ3_XXS (it does
    not today; `weights.cpp:282` restricts to IQ4_NL/BF16).

## 5. Additional HIP dispatch sites (qwen kernels, Phase 3 full route)

IQ4_NL/IQ4_XS/IQ3_S switch points to mirror if the shared qwen HIP route is
enabled (from the exploration; verify before editing):
`hip/kernels/prefill_fp16.hip` (54-58, 247-251, 287, 343, 361-364, 399-402,
458, 487, 523, 531-534, 559, 581, 772-785, 844-870, 886-887),
`hip/kernels/prefill_quant_gemm.hip` (1072-1096, 1119, 1254, 1407-1416,
1484-1493, 1712-1719), `hip/kernels/prefill_quant_wave64.hip` (17, 35-39),
`hip/kernels/small_batch_wave64.hip` (14-19, 78-109, 152, 193, 476-489,
526-527, 554-558, 588-625), `hip/kernels/small_batch_gemm.hpp` (442, 475),
`hip/kernels/swiglu.hip` (542-570), `hip/kernels/gemv_quant.hip` (182-183),
`hip/batched_decode.cpp` (55-59).

## 6. Verification commands (per AGENTS.md)

```sh
# Formatting before any commit
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py

# CPU checks (Phases 1-3)
nix develop -c cmake --preset cpu-test
nix develop -c cmake --build --preset pr          # hosted contract suite
nix develop -c ctest --preset cpu-test -R 'ggml_dequant|gemm_route|gguf_reader' --output-on-failure

# GPU kernel parity (Phase 4)
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target qwen_q4kxl_quant_ops_test
nix develop -c ctest --preset gpu-fast -R 'qwen.q4kxl_quant_ops' --output-on-failure

# End-to-end: model load + quality (do not treat a missing-model skip as a pass)
bash build_rocm.sh && bash run_rocm.sh
```

## 7. Decisions to validate before implementing

1. **Tier for IQ3_XXS**: quantized-only (enum+size+whitelist, CPU oracle
   dequant, GPU via vendored mmq vector path) vs full qwen shared-GEMM route
   (Phase 3 kernel switches). Quantized-only is the minimal path that makes
   `run_rocm.sh` work; the mmq device code for the expert vector path is
   already vendored.
2. **Grid table duplication**: `kIq3xxsGrid` in `ggml_dequant.cpp` vs the
   vendored `iq3xxs_grid` — pick one canonical copy (repo rule: one canonical
   definition per behavior).
3. **Sub-block scale semantics**: the 4-bit `ls` (`aux32 >> 28`) and sign-bit
   unpacking must be transcribed exactly from `vecdotq.hpp:1064-1095`;
   validate CPU decode against llama.cpp's `dequantize_iq3_xxs` on real shard
   bytes before trusting quality.
4. **Silent-zero default** in `cpu_ops.cpp:82` — consider failing loudly for
   unknown types instead, so a missing case can never mask a quality bug.
 5. `run_rocm.sh` defines `MTP` but never passes it (line 20) — unrelated to
    this fix, note for cleanup.

## 8. Performance expectations

Production inference stays **GPU/HIP-accelerated**; nothing in this plan moves
hot compute to the CPU.

- Weights are uploaded to the device **as-quantized** — the upload path never
  dequantizes (`kernels/rocm/device_model.cpp:46-79`). IQ3_XXS bytes go up
  unchanged.
- The IQ3_XXS **device kernels already exist** in the vendored mmq layer:
  `vec_dot_iq3_xxs_q8_1` (`mmq/vecdotq.hpp:1064-1095`), MMQ tile traits
  (`mmq.hpp:3356-3357`), `DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS)` (`mmq.hpp:4309`).
  Phase 4 only instantiates them and opens the type guard.
- The CPU `DequantizeIQ3_XXS`/`DotProductIQ3_XXS` (Phase 1) feed only the CPU
  reference oracle (`cpu_ops.cpp`) and unit tests — never the serving path.

**The real caveat: WMMA fast path excludes IQ3_XXS.** The batched WMMA expert
route is gated to Q4_K/Q5_K (gate/up) and Q5_1/Q8_0 (down) at
`executor.cpp:1587-1593` (tiled dispatch 991-1006). IQ3_XXS experts therefore
run on the **vector mmq (DP4A) route**, the same kernel family llama.cpp uses
for IQ types on gfx1151:

| Workload | Bound | Expectation vs Q4_K experts |
|---|---|---|
| Decode (top-k 10/512 experts) | memory bandwidth | on par or better — IQ3_XXS is 3.0625 bpw vs 4.5 bpw, ~30% fewer expert bytes read |
| Prefill (many tokens/expert) | compute | likely slower — DP4A vec-dot with grid lookup + sign unpacking, no WMMA tile acceleration |

**Is a WMMA fast path possible for IQ3_XXS? Yes — three levels, none impossible.**
The WMMA exclusion is not a hardware limit: IQ3_XXS values come from a
codebook lookup (`iq3xxs_grid[q3[l]]` yields four magnitude bytes, then
per-element sign XOR, `vecdotq.hpp:1076-1089`) instead of the linear nibble
math of K-quants, so nobody has written the dequant-to-fragment routine yet.

- **Level 0 — DP4A mmq (already exists).** The vendored layer *is* a GPU
  path: `vec_dot_iq3_xxs_q8_1` + tile traits + `DECL_MMQ_CASE` are complete
  device code. Phase 4 is pure C++ wiring (instantiate
  `mul_mat_q_case<GGML_TYPE_IQ3_XXS>`, open the guard). Zero new kernels.
- **Level 1 — IQ3_XXS WMMA fast path (feasible, real kernel work).**
  Magnitude bytes + signs convert cleanly to fp16/bf16 WMMA fragments (or
  int8 WMMA with split accumulation); the only novel step versus K-quants is
  the grid lookup, which stages into LDS like DS4 already does. In-repo
  precedents for both halves: WMMA quantized GEMM structure in
  `hip/kernels/prefill_quant_gemm.hip` and `swiglu.hip`; bespoke device IQ
  kernel with LDS-staged grid table in
  `deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_iq2_gate.hip.hpp`
  (`hip_iq2xxs_grid` staging, `dev_iq2_dp4a_8`, `dev_iq2_i8x8_lut`). Needs a
  quality-oracle pass per the optimize-kernel skill.
- **Level 2 — requantize at upload (pragmatic middle ground).** A one-shot
  device kernel converts IQ3_XXS → Q8_0 during `DeviceModel::Upload`;
  experts then flow through the existing Q8_0 WMMA/tiled paths untouched.
  Cost: expert footprint grows 3.06 → 8 bpw (~2.6× more expert bytes),
  surrendering the decode bandwidth win. No new GEMM kernels.

**Recommendation:** ship Level 0 first and measure before paying for WMMA —
the grid lookup amortizes well under DP4A and prefill may simply be fine.
Profile with `tools/prof/prof.py` (pipeline/wall time),
`tools/bench/build.sh` (standalone kernel experiments), and the
`benchmark-model` skill (end-to-end cells vs the llama.cpp reference).
If prefill disappoints, Level 2 is a small, low-risk addition; Level 1 is the
proper fix. (Avoidance option, no code at all: serve a Q4_K_XL/Q8_0
quantization whose experts already pass the whitelist.)

Decision rule: treat the DP4A route as the production default only if the
measured prefill cost is acceptable for the target workload; otherwise prefer
Level 2 for prefill-heavy serving and revisit Level 1.

