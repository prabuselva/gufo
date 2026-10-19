# Tokenizer contracts are CPU-only and run in the routine suite.
add_custom_target(nemotron_h_moe_tests)

add_executable(nemotron_h_moe_tokenizer_test tokenizer_test.cpp)
target_link_libraries(nemotron_h_moe_tokenizer_test PRIVATE
  gufo_nemotron_h_moe)
add_test(NAME nemotron_h_moe.tokenizer COMMAND nemotron_h_moe_tokenizer_test)
set_tests_properties(nemotron_h_moe.tokenizer PROPERTIES
  LABELS "nemotron_h_moe;cpu")
add_dependencies(nemotron_h_moe_tests nemotron_h_moe_tokenizer_test)

# Scalar oracle contracts: a synthetic miniature trunk always runs; the real
# 24GB artifact section skips (77) unless GUFO_NEMOTRON_H_MOE_GGUF points at
# it. Built only when the reference library is requested.
add_executable(nemotron_h_moe_reference_test EXCLUDE_FROM_ALL
  reference_test.cpp)
target_link_libraries(nemotron_h_moe_reference_test PRIVATE
  gufo_nemotron_h_moe_reference)
set_source_files_properties(reference_test.cpp PROPERTIES
  COMPILE_OPTIONS "-O2")
add_test(NAME nemotron_h_moe.reference COMMAND nemotron_h_moe_reference_test)
set_tests_properties(nemotron_h_moe.reference PROPERTIES
  LABELS "nemotron_h_moe;cpu;external-model;slow" SKIP_RETURN_CODE 77)

# Vendored MMQ/MMVQ contracts on gfx1151: synthetic IQ2_XXS/NVFP4 matrices
# through the dense, dense-vec and routed moe-vec tiers against the scalar
# oracle's dequantizers.
if (ENGINE_ENABLE_HIP)
  add_executable(nemotron_h_moe_rocm_mmq_test EXCLUDE_FROM_ALL
    rocm_mmq_test.cpp)
  target_link_libraries(nemotron_h_moe_rocm_mmq_test PRIVATE
    gufo_nemotron_h_moe_mmq gufo_nemotron_h_moe_reference)
  add_test(NAME nemotron_h_moe.rocm_mmq COMMAND nemotron_h_moe_rocm_mmq_test)
  set_tests_properties(nemotron_h_moe.rocm_mmq PROPERTIES
    LABELS "nemotron_h_moe;hip;gpu;gfx1151")
  add_dependencies(nemotron_h_moe_tests nemotron_h_moe_rocm_mmq_test)

  # Fused elementwise kernel contracts against the scalar cpu_* oracle.
  add_executable(nemotron_h_moe_rocm_ops_test EXCLUDE_FROM_ALL
    rocm_ops_test.cpp)
  target_link_libraries(nemotron_h_moe_rocm_ops_test PRIVATE
    gufo_nemotron_h_moe_kernels gufo_nemotron_h_moe_reference)
  add_test(NAME nemotron_h_moe.rocm_ops COMMAND nemotron_h_moe_rocm_ops_test)
  set_tests_properties(nemotron_h_moe.rocm_ops PROPERTIES
    LABELS "nemotron_h_moe;hip;gpu;gfx1151")
  add_dependencies(nemotron_h_moe_tests nemotron_h_moe_rocm_ops_test)
endif()
