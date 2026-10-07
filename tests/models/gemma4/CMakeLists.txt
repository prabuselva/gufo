# Metadata checks are CPU-only and run in the routine suite. Operator and
# model checks are added as their kernels land.
add_custom_target(gemma4_tests)

add_executable(gemma4_config_test config_test.cpp)
target_link_libraries(gemma4_config_test PRIVATE gufo_gemma4)
add_test(NAME gemma4.config COMMAND gemma4_config_test)
set_tests_properties(gemma4.config PROPERTIES LABELS "gemma4;cpu")
add_dependencies(gemma4_tests gemma4_config_test)

# Smart-resize policy and CHW normalization contract (CPU-only, no model file).
add_executable(gemma4_vision_prompt_test vision_prompt_test.cpp)
target_link_libraries(gemma4_vision_prompt_test PRIVATE gufo_gemma4)
add_test(NAME gemma4.vision_prompt COMMAND gemma4_vision_prompt_test)
set_tests_properties(gemma4.vision_prompt PROPERTIES LABELS "gemma4;cpu")
add_dependencies(gemma4_tests gemma4_vision_prompt_test)

# Binds the real artifacts (headers only) to prove every tensor name and shape
# resolves. Skips (77) unless GUFO_GEMMA4_GGUF points at a Gemma-4 GGUF; the
# draft checks additionally need GUFO_GEMMA4_MTP_GGUF.
add_executable(gemma4_weights_test weights_test.cpp)
target_link_libraries(gemma4_weights_test PRIVATE gufo_gemma4)
add_test(NAME gemma4.weights COMMAND gemma4_weights_test)
set_tests_properties(gemma4.weights PROPERTIES
  LABELS "gemma4;cpu;external-model" SKIP_RETURN_CODE 77)
add_dependencies(gemma4_tests gemma4_weights_test)

# Tokenizer parity against golden vectors captured from the llama.cpp fork.
# The synthetic merge-engine checks always run; the golden vectors skip (77)
# without GUFO_GEMMA4_GGUF.
add_executable(gemma4_tokenizer_test tokenizer_test.cpp)
target_link_libraries(gemma4_tokenizer_test PRIVATE gufo_gemma4)
add_test(NAME gemma4.tokenizer COMMAND gemma4_tokenizer_test)
set_tests_properties(gemma4.tokenizer PROPERTIES
  LABELS "gemma4;cpu;external-model" SKIP_RETURN_CODE 77)
add_dependencies(gemma4_tests gemma4_tokenizer_test)

# Chat-template renderer goldens (byte-exact vs the pinned Jinja template)
# always run; the artifact hash check skips its tail without GUFO_GEMMA4_GGUF.
add_executable(gemma4_chat_template_test chat_template_test.cpp)
target_link_libraries(gemma4_chat_template_test PRIVATE gufo_gemma4)
add_test(NAME gemma4.chat_template COMMAND gemma4_chat_template_test)
set_tests_properties(gemma4.chat_template PROPERTIES LABELS "gemma4;cpu")
add_dependencies(gemma4_tests gemma4_chat_template_test)

# Runs the scalar oracle over the real artifact (a few tokens). Built only when
# the reference library is requested, and skips (77) without the GGUF.
add_executable(gemma4_reference_test EXCLUDE_FROM_ALL reference_test.cpp)
target_link_libraries(gemma4_reference_test PRIVATE gufo_gemma4_reference)
set_source_files_properties(reference_test.cpp PROPERTIES COMPILE_OPTIONS "-O2")
add_test(NAME gemma4.reference COMMAND gemma4_reference_test)
set_tests_properties(gemma4.reference PROPERTIES
  LABELS "gemma4;cpu;external-model;slow" SKIP_RETURN_CODE 77)

# Binds the real mmproj and runs the scalar vision oracle over a small image.
# Skips (77) without GUFO_GEMMA4_MMPROJ_GGUF.
add_executable(gemma4_vision_reference_test EXCLUDE_FROM_ALL
  vision_reference_test.cpp)
target_link_libraries(gemma4_vision_reference_test PRIVATE
  gufo_gemma4_reference)
set_source_files_properties(vision_reference_test.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")
add_test(NAME gemma4.vision_reference COMMAND gemma4_vision_reference_test)
set_tests_properties(gemma4.vision_reference PROPERTIES
  LABELS "gemma4;cpu;external-model;slow" SKIP_RETURN_CODE 77)

if (ENGINE_ENABLE_HIP)
  # Kernel-tier checks: synthetic tensors at the artifact geometry through the
  # GEMV/GEMM tiers, the fused routing operators and the routed F16 MoE
  # pipeline, each against a double-precision host reference. No model
  # artifact required.
  add_executable(gemma4_rocm_kernels_test rocm_kernels_test.cpp)
  target_link_libraries(gemma4_rocm_kernels_test PRIVATE
    gufo_gemma4_kernels gufo_core)
  add_test(NAME gemma4.rocm_kernels COMMAND gemma4_rocm_kernels_test)
  set_tests_properties(gemma4.rocm_kernels PROPERTIES
    LABELS "gemma4;hip;gpu;gfx1151")
  add_dependencies(gemma4_tests gemma4_rocm_kernels_test)

  # Standalone prefill-GEMM microbenchmark: the production int8 MMQ path
  # against the model-private binary16 WMMA path (gemma4 DenseF16Gemm) on the
  # six dense projections the trunk runs at pp2048. Not a CTest case; run it
  # under tools/bench/gpu_exclusive.sh. See docs/models/gemma4/OPTIMIZATION_DIFF.md.
  add_executable(gemma4_dense_f16_bench EXCLUDE_FROM_ALL dense_f16_bench.cpp)
  target_link_libraries(gemma4_dense_f16_bench PRIVATE
    gufo_gemma4_kernels gufo_qwen38_flash_next_mmq gufo_core)

  # End-to-end forward parity on the real artifact: prefill and decode logits
  # against the scalar oracle (<= 5e-3 abs, identical argmax). Skips (77)
  # without GUFO_GEMMA4_GGUF; needs the full GPU and ~28 GiB of VRAM.
  add_executable(gemma4_rocm_forward_test rocm_forward_test.cpp)
  target_link_libraries(gemma4_rocm_forward_test PRIVATE
    gufo_gemma4_executor gufo_gemma4_reference gufo_core)
  add_test(NAME gemma4.rocm_forward COMMAND gemma4_rocm_forward_test)
  set_tests_properties(gemma4.rocm_forward PROPERTIES
    LABELS "gemma4;hip;gpu;gfx1151;external-model;slow" SKIP_RETURN_CODE 77)

  # MTP draft forward parity on the real trunk plus its sidecar: the GPU draft
  # step against the scalar oracle over a shared hidden input (scale-relative
  # logit bound, identical argmax, deterministic, chained h_next). Skips (77)
  # without GUFO_GEMMA4_GGUF and GUFO_GEMMA4_MTP_GGUF; needs the full GPU.
  add_executable(gemma4_mtp_test gemma4_mtp_test.cpp)
  target_link_libraries(gemma4_mtp_test PRIVATE
    gufo_gemma4_executor gufo_gemma4_reference gufo_core)
  add_test(NAME gemma4.mtp COMMAND gemma4_mtp_test)
  set_tests_properties(gemma4.mtp PROPERTIES
    LABELS "gemma4;hip;gpu;gfx1151;external-model;slow" SKIP_RETURN_CODE 77)

  # Speculative decode parity on the real trunk plus its sidecar: greedy output
  # with MTP enabled is token-for-token identical to greedy without it, and the
  # speculative run actually accepts drafts. Skips (77) without GUFO_GEMMA4_GGUF
  # and GUFO_GEMMA4_MTP_GGUF; needs the full GPU.
  add_executable(gemma4_speculative_test gemma4_speculative_test.cpp)
  target_link_libraries(gemma4_speculative_test PRIVATE
    gufo_gemma4_engine gufo_core)
  add_test(NAME gemma4.speculative COMMAND gemma4_speculative_test)
  set_tests_properties(gemma4.speculative PROPERTIES
    LABELS "gemma4;hip;gpu;gfx1151;external-model;slow" SKIP_RETURN_CODE 77)

  # Vision tower device parity on the real mmproj: the ROCm encoder against the
  # scalar oracle over a fixed 96x96 image (scale-relative bound). Skips (77)
  # without GUFO_GEMMA4_MMPROJ_GGUF; needs the GPU.
  add_executable(gemma4_vision_encoder_test EXCLUDE_FROM_ALL
    vision_encoder_test.cpp)
  target_link_libraries(gemma4_vision_encoder_test PRIVATE
    gufo_gemma4_executor gufo_gemma4_reference gufo_core)
  add_test(NAME gemma4.vision_encoder COMMAND gemma4_vision_encoder_test)
  set_tests_properties(gemma4.vision_encoder PROPERTIES
    LABELS "gemma4;hip;gpu;gfx1151;external-model;slow" SKIP_RETURN_CODE 77)
endif()