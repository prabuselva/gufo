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
