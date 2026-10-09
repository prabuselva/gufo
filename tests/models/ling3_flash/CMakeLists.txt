# Metadata checks are CPU-only and run in the routine suite. Operator and
# model checks are added as their kernels land.
add_custom_target(ling3_flash_tests)

add_executable(ling3_flash_config_test config_test.cpp)
target_link_libraries(ling3_flash_config_test PRIVATE gufo_ling3_flash)
add_test(NAME ling3_flash.config COMMAND ling3_flash_config_test)
set_tests_properties(ling3_flash.config PROPERTIES LABELS "ling3_flash;cpu")
add_dependencies(ling3_flash_tests ling3_flash_config_test)

# Binds the real artifact (headers only) to prove every tensor name and shape
# resolves. Skips (77) unless GUFO_LING3_FLASH_GGUF points at a Ling 3.0 Flash
# GGUF (the first shard of a split file).
add_executable(ling3_flash_weights_test weights_test.cpp)
target_link_libraries(ling3_flash_weights_test PRIVATE gufo_ling3_flash)
add_test(NAME ling3_flash.weights COMMAND ling3_flash_weights_test)
set_tests_properties(ling3_flash.weights PROPERTIES
  LABELS "ling3_flash;cpu;external-model" SKIP_RETURN_CODE 77)
add_dependencies(ling3_flash_tests ling3_flash_weights_test)

# Runs the scalar oracle over the real artifact (a few tokens). Built only when
# the reference library is requested, and skips (77) without the GGUF.
add_executable(ling3_flash_reference_test EXCLUDE_FROM_ALL reference_test.cpp)
target_link_libraries(ling3_flash_reference_test PRIVATE
  gufo_ling3_flash_reference)
set_source_files_properties(reference_test.cpp PROPERTIES COMPILE_OPTIONS "-O2")
add_test(NAME ling3_flash.reference COMMAND ling3_flash_reference_test)
set_tests_properties(ling3_flash.reference PROPERTIES
  LABELS "ling3_flash;cpu;external-model;slow" SKIP_RETURN_CODE 77)

if (ENGINE_ENABLE_HIP)
  # GEMV checks: synthetic quantized matrices through the GEMV tier against a
  # CPU reference that dequantizes with the canonical ggml dequantizers
  # (gufo_core) and dots in double. No model artifact required.
  add_executable(ling3_flash_rocm_gemv_test rocm_gemv_test.cpp)
  target_link_libraries(ling3_flash_rocm_gemv_test PRIVATE
    gufo_ling3_flash_kernels gufo_core)
  add_test(NAME ling3_flash.rocm_gemv COMMAND ling3_flash_rocm_gemv_test)
  set_tests_properties(ling3_flash.rocm_gemv PROPERTIES
    LABELS "ling3_flash;hip;gpu;gfx1151")
  add_dependencies(ling3_flash_tests ling3_flash_rocm_gemv_test)

  # Fused-operator checks: synthetic tensors through every fused kernel
  # against a host reference mirroring the scalar oracle formulas.
  add_executable(ling3_flash_rocm_ops_test rocm_ops_test.cpp)
  target_link_libraries(ling3_flash_rocm_ops_test PRIVATE
    gufo_ling3_flash_kernels)
  add_test(NAME ling3_flash.rocm_ops COMMAND ling3_flash_rocm_ops_test)
  set_tests_properties(ling3_flash.rocm_ops PROPERTIES
    LABELS "ling3_flash;hip;gpu;gfx1151")
  add_dependencies(ling3_flash_tests ling3_flash_rocm_ops_test)

  # Absorbed-MLA decode checks: synthetic latent caches through the split-KV
  # partial+combine kernels against a host double reference.
  add_executable(ling3_flash_rocm_attention_test rocm_attention_test.cpp)
  target_link_libraries(ling3_flash_rocm_attention_test PRIVATE
    gufo_ling3_flash_kernels)
  add_test(NAME ling3_flash.rocm_attention COMMAND
    ling3_flash_rocm_attention_test)
  set_tests_properties(ling3_flash.rocm_attention PROPERTIES
    LABELS "ling3_flash;hip;gpu;gfx1151")
  add_dependencies(ling3_flash_tests ling3_flash_rocm_attention_test)

  # End-to-end forward pass: the GPU executor's trunk, hidden stream and MTP
  # logits must agree with the scalar oracle over the real artifact. Skips (77)
  # without the GGUF.
  add_executable(ling3_flash_rocm_forward_test EXCLUDE_FROM_ALL
    rocm_forward_test.cpp)
  target_link_libraries(ling3_flash_rocm_forward_test PRIVATE
    gufo_ling3_flash_executor gufo_ling3_flash_reference)
  add_test(NAME ling3_flash.rocm_forward COMMAND ling3_flash_rocm_forward_test)
  set_tests_properties(ling3_flash.rocm_forward PROPERTIES
    LABELS "ling3_flash;hip;gpu;gfx1151;external-model;slow"
    SKIP_RETURN_CODE 77)

  # CPU-vs-GPU greedy decode: the token streams must agree exactly, which is
  # the quality gate the forward test cannot provide. Skips (77) without the
  # GGUF.
  add_executable(ling3_flash_rocm_inference_test EXCLUDE_FROM_ALL
    rocm_inference_test.cpp)
  target_link_libraries(ling3_flash_rocm_inference_test PRIVATE
    gufo_ling3_flash_executor gufo_ling3_flash_reference)
  add_test(NAME ling3_flash.rocm_inference COMMAND
    ling3_flash_rocm_inference_test)
  set_tests_properties(ling3_flash.rocm_inference PROPERTIES
    LABELS "ling3_flash;hip;gpu;gfx1151;external-model;slow"
    SKIP_RETURN_CODE 77)

  # Greedy quality gate over real text: parity cannot catch a shared formula
  # error, so the artifact itself must produce a coherent continuation against
  # the reference server's ground truth. Skips (77) without the GGUF.
  add_executable(ling3_flash_rocm_quality_test EXCLUDE_FROM_ALL
    rocm_quality_test.cpp)
  target_link_libraries(ling3_flash_rocm_quality_test PRIVATE
    gufo_ling3_flash_engine)
  add_test(NAME ling3_flash.rocm_quality COMMAND
    ling3_flash_rocm_quality_test)
  set_tests_properties(ling3_flash.rocm_quality PROPERTIES
    LABELS "ling3_flash;hip;gpu;gfx1151;external-model;slow"
    SKIP_RETURN_CODE 77)
endif()