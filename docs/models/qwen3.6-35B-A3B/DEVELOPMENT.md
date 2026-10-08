# Qwen3.6-35B-A3B (qwen35moe) — Development Log

Status: **RESOLVED.** The GPU executor was correct all along. The apparent
"GPU produces garbled text" bug was a device-memory read bug in the *test* and
the *engine*: both read the executor's device logits buffer with
`std::copy_n` (a raw host read of a `hipMalloc`'d device pointer) instead of
`hipMemcpy`. On this system that read returned stale/garbage data, so the
first "generated" token was actually the last prompt token and the stream
degenerated. Replacing the read with `hipMemcpy` makes the CPU/GPU token
streams match exactly and the CLI produce coherent text.

## Goal

Implement Qwen3.6-35B-A3B (GGUF arch `qwen35moe`) in Gufo on Strix Halo
gfx1151. The model is a hybrid: Gated DeltaNet (linear attention) layers +
gated grouped-query attention layers + MoE + an MTP draft block. The
implementation is isolated from Qwen27B / flash-next: new dir
`src/models/qwen36_a3b`, own kernels, zero edits to `src/models/qwen` or
flash-next.

## Model geometry (from the artifact)

- hidden 2048, 16 query heads / 2 KV heads, head_dim 256, rotary 64,
  rope_theta 1e7, rms_eps 1e-6.
- 40 trunk layers + 1 MTP block (layer index 40, full-attention).
- `IsLinearLayer(il) = (il+1) % 4 != 0` → 30 GDN (linear) layers + 10 GQA
  (full) layers at il 3,7,11,15,19,23,27,31,35,39.
- GDN: ssm_conv_kernel 4, ssm_head_dim 128, ssm_num_k_heads 16,
  ssm_num_v_heads 32, ssm_inner 4096. Derived: SsmConvChannels 8192,
  SsmKeyDim 2048, SsmValueDim 4096.
- GQA: AttentionQDim 4096, AttentionKvDim 512.
- MoE: 256 experts, top-8, expert_ff 512, shared_expert_ff 512.
- vocab 248320.
- Weights are Q8_0 + F32 + BF16 only (verified by byte-size). They stay
  quantized on the GPU (dequant-to-float ≈ 140 GB would not fit in 79 GB
  free) → quantized GEMV.

## Build / test environment (required)

```sh
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH   # else libamdhip64.so.7 not found
export GUFO_QWEN36_A3B_GGUF=$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
```

Configure (preset alone fails with "Failed to find HIP root"):

```sh
rm -rf build/gpu-test
export CMAKE_PREFIX_PATH=/opt/rocm
cmake --preset gpu-test -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
```

CMake rejects `hipcc`; use `/opt/rocm/lib/llvm/bin/clang++` (AMD clang
23.0.0). Build a target:

```sh
cmake --build build/gpu-test --target <target> --parallel 4
```

Real GGUF: 36.41 GiB at the path above.

## What is done (committed)

- `173ba67` M1: config loader + config_test.
- `b4f853e` M2: weight loader (raw-block binding + shape validation) +
  weights_test.
- `9500af3` M3: CPU scalar oracle `cpu_ops.{hpp,cpp}` +
  `reference.{hpp,cpp}` (ReferenceModel Step/MtpStep), lib
  `gufo_qwen36_a3b_reference` (-O2), env-gated `reference_test.cpp`.
- `fafd515` M4: ROCm fused kernels (`kernels.hpp` + `kernels.hip.cpp`, 11
  kernels) + 3 GPU tests + `hip_test.hpp`. All pass via ctest.
- `bd2a620` GEMV: `gemv.{hpp,hip.cpp}` (Q8_0/F32/BF16 mat-vec, one warp/row,
  `__shfl_down`) + `rocm_gemv_test.cpp`.
- `dee47d7` GPU executor + end-to-end forward test (11 files, 1132 insertions).

### Uncommitted working-tree changes

- `src/models/qwen36_a3b/engine.{hpp,cpp}` — Model/Session wrapper around the
  executor (no vision, no ngram, no speculative MTP).
- `src/models/qwen36_a3b/CMakeLists.txt` — `gufo_qwen36_a3b_engine` target.
- `CMakeLists.txt` (top-level) — `gufo_llm_cli` links
  `gufo_qwen36_a3b_engine`.
- `src/cli/prompt/prompt.cpp` — `qwen35moe` wiring (LoadQwen36A3BModel,
  GenerateQwen36A3BResponse, architecture check before `qwen4exp`).
- `tests/models/qwen36_a3b/rocm_inference_test.cpp` — NEW CPU+GPU greedy
  decode comparison on a real prompt.
- `tests/models/qwen36_a3b/CMakeLists.txt` — registers
  `qwen36_a3b_rocm_inference_test`.

## The bug (root cause)

`qwen36_a3b_rocm_inference_test "The capital of France is" 4` originally gave:

```
prompt: [760 The, 6511 capital, 314 of, 9338 France, 369 is]
CPU tokens: 11751 11 264 3177   -> "Paris, a city"   (CORRECT)
GPU tokens: 369 11751 11751 13  -> "is Paris Paris." (WRONG)
```

The GPU's first "generated" token was 369 ("is"), the **last prompt token**,
then it degenerated into repetition. This looked like a position / KV-cache /
logits-update bug in the executor, but the executor was correct.

### Root cause: raw host read of a device pointer

`Executor::logits()` returns `logits_`, a `hipMalloc`'d **device** pointer.
Two call sites read it with `std::copy_n` (a plain host `memcpy` of the
pointer) instead of `hipMemcpy`:

- `tests/models/qwen36_a3b/rocm_inference_test.cpp` (prefill + decode read)
- `src/models/qwen36_a3b/engine.cpp` `Session::Sync` / `Evaluate` /
  `DecodeStep` (the production CLI path)

On this system that host read of device memory returned stale/garbage values,
so the "GPU logits" were not the values the GPU actually computed. The forward
test was unaffected because it downloads logits with `hipMemcpy`
(`Download()`), which is why it passed while the inference test and the CLI
failed.

### Fix

Replace every `std::copy_n(executor->logits(), n, host.begin())` with
`hipMemcpy(host, executor->logits(), n * sizeof(float), hipMemcpyDeviceToHost)`.
Applied in both files above.

### Verification

- `qwen36_a3b_rocm_inference_test "The capital of France is" 16`: all 5
  prefill steps match (worst relative ≤ 2.8e-3, argmax identical) and the 16
  generated tokens match the CPU exactly.
- `gufo prompt -m <GGUF> -p "The capital of France is" -n 32 -t 0 --raw`
  → "Paris, a city renowned for its rich history, culture, and iconic
  landmarks. Situated in the north-central part of the country, along the
  Seine River,"
- `gufo prompt -m <GGUF> -p "What is the capital of France? ..." -n 64 -t 0
  --think off` → "The capital of France is Paris."

## Next steps

The blocking bug is fixed and verified. Remaining to land the model:

1. Run the full `qwen36_a3b` GPU suite to confirm nothing else regressed:
   `ctest --test-dir build/gpu-test -R '^qwen36_a3b\.' --output-on-failure`.
2. Commit the fix: `src/models/qwen36_a3b/engine.cpp` (3× `hipMemcpy`),
   `tests/models/qwen36_a3b/rocm_inference_test.cpp` (per-step prefill
   compare + `hipMemcpy`), plus the uncommitted engine/CLI wiring.
3. Optional hardening: add a `Executor::DownloadLogits(float*, size_t)`
   helper so call sites cannot regress to a raw host read of the device
   buffer again.

## Key implementation notes

- **GEMV** (`gemv.hpp`): `void Gemv(const void* base, GemvType type,
  uint32_t rows, uint32_t cols, size_t row_bytes, const float* x, float* out,
  hipStream_t stream)` — one warp/row, float accumulation. `EmbedRow`
  dequantizes one row. `GemvType` is a local enum `{kQ8_0,kF32,kBF16}` (not
  `core::GgmlType`) to keep the C++17 kernels target free of C++20
  `std::span`.
- **11 fused kernels** (`kernels.hpp`): RmsNormRows, Rope, SplitQGate, Swiglu,
  SigmoidMul, Add, RouterTopK, MoeEpilogue, GdnConv, GdnNormQk, GdnDelta,
  GdnOutNorm, AttentionDecode. All raw pointers + hipStream_t, namespace
  `gufo::models::qwen36_a3b::rocm`. RmsNormRows is safe in-place.
- **GdnConv** shift loop was fixed (OOB read): `for (t = 0; t + 2 < kernel;
  ++t)`.
- **Executor** uses the default stream (nullptr) for all launches/copies;
  synchronous `hipMemcpy` for position upload and MoE id download. Position
  uploaded once per Step via a member host buffer.
- **MoE expert loop is host-side**: RouterTopK on GPU, download 8 expert ids
  (one sync per MoE layer), loop on host launching per-expert
  Gemv+Swiglu+Gemv, then MoeEpilogue.
- **Tolerance 2e-2** in the forward test: the reference `MatVec` accumulates
  in double; the GPU GEMV accumulates in float (one warp/row). With
  cancellation in the 2048-term dot product the float partial sums drift
  ~1e-2 relative on the largest logits. The argmax must still match exactly.
- **gufo `GgmlType` is non-standard**: kF32=0, kF16=1, kQ4_0=2, kQ4_1=3,
  kQ5_0=6, kQ5_1=7, kQ8_0=8, kQ8_1=9, kQ2_K=10, kQ3_K=11, kQ4_K=12, kQ5_K=13,
  kQ6_K=14, kQ8_K=15, kIQ2_XXS=16, kIQ4_NL=20, kIQ3_S=21, kIQ4_XS=23, kI32=26,
  kBF16=30.
- **Q8_0 block is 34 bytes** (`{__half d; int8 qs[32]}`, no padding). A
  `struct alignas(8)` inflates it to 40 bytes and breaks indexing.
- **BF16 on host**: `static_cast<float>(hip_bfloat16)` fails in a host C++
  compile (device-only); use manual bit conversion. `__half` /
  `__float2half` / `__half2float` do work on host.
- LSP "file not found" / `__device__` unknown / `gufo`/`core` undeclared
  errors are environmental (no HIP/project include root in the LSP); the real
  check is the gpu-test compile.

## Relevant files

- `src/models/qwen36_a3b/kernels/rocm/executor.{hpp,cpp}` — Executor forward
  pass. **Verified correct** (the bug was in the callers reading its device
  logits, not here).
- `src/models/qwen36_a3b/kernels/rocm/device_model.{hpp,cpp}` —
  DeviceTensor/DeviceLayer/DeviceModel::Upload.
- `src/models/qwen36_a3b/kernels/rocm/kernels.hip.cpp` — 11 fused ops.
- `src/models/qwen36_a3b/kernels/rocm/gemv.{hpp,hip.cpp}` — GEMV + EmbedRow.
- `src/models/qwen36_a3b/weights.{hpp,cpp}` — weight loader.
- `src/models/qwen36_a3b/config.{hpp,cpp}` — config.
- `src/models/qwen36_a3b/cpu_ops.{hpp,cpp}` — CPU oracle (double accumulation).
- `src/models/qwen36_a3b/reference.{hpp,cpp}` — ReferenceModel ground truth
  (verified correct: produces sensible text).
- `src/models/qwen36_a3b/engine.{hpp,cpp}` — Model/Session wrapper. **Site of
  the bug** (3× `std::copy_n` device reads, now `hipMemcpy`).
- `tests/models/qwen36_a3b/rocm_forward_test.cpp` — GPU forward test (passes;
  downloads logits with `hipMemcpy`, which is why it never caught the bug).
- `tests/models/qwen36_a3b/rocm_inference_test.cpp` — CPU+GPU greedy decode
  comparison. **Site of the bug** (now `hipMemcpy` + per-step prefill
  compare); passes.
- `tests/models/qwen36_a3b/hip_test.hpp` — GPU test scaffolding.
- `src/cli/prompt/prompt.cpp` — CLI wiring.
- Reference (do not modify): `src/models/qwen38_flash_next/engine.{hpp,cpp}`,
  `src/models/qwen38_flash_next/kernels/rocm/*`.
- External reference: `~/projects/llm/llama.cpp/llama.cpp/` (qwen3next /
  qwen35moe GDN + MoE + MTP).