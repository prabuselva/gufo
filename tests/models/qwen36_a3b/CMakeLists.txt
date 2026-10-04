# Metadata checks are CPU-only and run in the routine suite. Operator and
# model checks are added as their kernels land.
add_custom_target(qwen36_a3b_tests)

add_executable(qwen36_a3b_config_test config_test.cpp)
target_link_libraries(qwen36_a3b_config_test PRIVATE gufo_qwen36_a3b)
add_test(NAME qwen36_a3b.config COMMAND qwen36_a3b_config_test)
set_tests_properties(qwen36_a3b.config PROPERTIES LABELS "qwen36_a3b;cpu")
add_dependencies(qwen36_a3b_tests qwen36_a3b_config_test)

# Binds the real artifact (headers only) to prove every tensor name and shape
# resolves. Skips (77) unless GUFO_QWEN36_A3B_GGUF points at a Qwen3.6 GGUF.
add_executable(qwen36_a3b_weights_test weights_test.cpp)
target_link_libraries(qwen36_a3b_weights_test PRIVATE gufo_qwen36_a3b)
add_test(NAME qwen36_a3b.weights COMMAND qwen36_a3b_weights_test)
set_tests_properties(qwen36_a3b.weights PROPERTIES
  LABELS "qwen36_a3b;cpu;external-model" SKIP_RETURN_CODE 77)
add_dependencies(qwen36_a3b_tests qwen36_a3b_weights_test)

# Runs the scalar oracle over the real artifact (a few tokens). Built only when
# the reference library is requested, and skips (77) without the GGUF.
add_executable(qwen36_a3b_reference_test EXCLUDE_FROM_ALL reference_test.cpp)
target_link_libraries(qwen36_a3b_reference_test PRIVATE
  gufo_qwen36_a3b_reference)
set_source_files_properties(reference_test.cpp PROPERTIES COMPILE_OPTIONS "-O2")
add_test(NAME qwen36_a3b.reference COMMAND qwen36_a3b_reference_test)
set_tests_properties(qwen36_a3b.reference PROPERTIES
  LABELS "qwen36_a3b;cpu;external-model;slow" SKIP_RETURN_CODE 77)

if (ENGINE_ENABLE_HIP)
  # Fused-operator checks: synthetic activations through each kernel against a
  # CPU reference of the scalar oracle's formulas. No model artifact required.
  # gufo_core supplies the canonical Q4_K/Q5_K/Q6_K dequantizers the quantized
  # checks use as their ground truth.
  foreach(name IN ITEMS rocm_ops rocm_gdn rocm_attention rocm_gemv rocm_gemm)
    add_executable(qwen36_a3b_${name}_test ${name}_test.cpp)
    target_link_libraries(qwen36_a3b_${name}_test PRIVATE
      gufo_qwen36_a3b_kernels gufo_core)
    add_test(NAME qwen36_a3b.${name} COMMAND qwen36_a3b_${name}_test)
    set_tests_properties(qwen36_a3b.${name} PROPERTIES
      LABELS "qwen36_a3b;hip;gpu;gfx1151")
    add_dependencies(qwen36_a3b_tests qwen36_a3b_${name}_test)
  endforeach()

  # End-to-end forward pass: the GPU executor's trunk and MTP logits must agree
  # with the scalar oracle over the real artifact. Skips (77) without the GGUF.
  add_executable(qwen36_a3b_rocm_forward_test EXCLUDE_FROM_ALL
    rocm_forward_test.cpp)
  target_link_libraries(qwen36_a3b_rocm_forward_test PRIVATE
    gufo_qwen36_a3b_executor gufo_qwen36_a3b_reference)
  add_test(NAME qwen36_a3b.rocm_forward COMMAND qwen36_a3b_rocm_forward_test)
  set_tests_properties(qwen36_a3b.rocm_forward PROPERTIES
    LABELS "qwen36_a3b;hip;gpu;gfx1151;external-model;slow"
    SKIP_RETURN_CODE 77)

  # CPU-vs-GPU greedy decode over a real text prompt: the token streams must
  # agree exactly and the decoded text is the quality gate the forward test
  # cannot provide. Skips (77) without the GGUF.
  add_executable(qwen36_a3b_rocm_inference_test EXCLUDE_FROM_ALL
    rocm_inference_test.cpp)
  target_link_libraries(qwen36_a3b_rocm_inference_test PRIVATE
    gufo_qwen36_a3b_executor gufo_qwen36_a3b_reference gufo_core)
  add_test(NAME qwen36_a3b.rocm_inference COMMAND qwen36_a3b_rocm_inference_test)
  set_tests_properties(qwen36_a3b.rocm_inference PROPERTIES
    LABELS "qwen36_a3b;hip;gpu;gfx1151;external-model;slow"
    SKIP_RETURN_CODE 77)
endif()