# Metadata checks are CPU-only and run in the routine suite. Operator and
# model checks are added as their kernels land.
add_custom_target(flint_tests)

add_executable(flint_config_test config_test.cpp)
target_link_libraries(flint_config_test PRIVATE gufo_flint)
add_test(NAME flint.config COMMAND flint_config_test)
set_tests_properties(flint.config PROPERTIES LABELS "flint;cpu")
add_dependencies(flint_tests flint_config_test)

add_executable(flint_tokenizer_test tokenizer_test.cpp)
target_link_libraries(flint_tokenizer_test PRIVATE gufo_flint)
add_test(NAME flint.tokenizer COMMAND flint_tokenizer_test)
set_tests_properties(flint.tokenizer PROPERTIES LABELS "flint;cpu")
add_dependencies(flint_tests flint_tokenizer_test)

# Weight-table binding runs against the real artifact when GUFO_FLINT_GGUF
# names it and skips (77) otherwise.
add_executable(flint_weights_test weights_test.cpp)
target_link_libraries(flint_weights_test PRIVATE gufo_flint)
add_test(NAME flint.weights COMMAND flint_weights_test)
set_tests_properties(flint.weights PROPERTIES
  LABELS "flint;cpu;external-model" SKIP_RETURN_CODE 77)
add_dependencies(flint_tests flint_weights_test)

# Runs the scalar oracle over the real artifact (a few tokens). Built only when
# the reference library is requested, and skips (77) without the GGUF.
add_executable(flint_reference_test EXCLUDE_FROM_ALL reference_test.cpp)
target_link_libraries(flint_reference_test PRIVATE gufo_flint_reference)
set_source_files_properties(reference_test.cpp PROPERTIES COMPILE_OPTIONS "-O2")
add_test(NAME flint.reference COMMAND flint_reference_test)
set_tests_properties(flint.reference PROPERTIES
  LABELS "flint;cpu;external-model;slow" SKIP_RETURN_CODE 77)

# Engine Model/Session checks run against the real artifact on the GPU
# (exclusive access; a few forwards) and skip (77) without GUFO_FLINT_GGUF.
if (ENGINE_ENABLE_HIP)
  add_executable(flint_engine_test EXCLUDE_FROM_ALL engine_test.cpp)
  target_link_libraries(flint_engine_test PRIVATE gufo_flint_engine)
  add_test(NAME flint.engine COMMAND flint_engine_test)
  set_tests_properties(flint.engine PROPERTIES
    LABELS "flint;gpu;external-model;slow" SKIP_RETURN_CODE 77)
endif()

# ROCm trunk parity against the scalar oracle over the real artifact; needs
# exclusive GPU access and skips (77) without GUFO_FLINT_GGUF.
if (ENGINE_ENABLE_HIP)
  add_executable(flint_rocm_forward_test EXCLUDE_FROM_ALL
    rocm_forward_test.cpp)
  target_link_libraries(flint_rocm_forward_test PRIVATE
    gufo_flint_executor gufo_flint_reference)
  set_source_files_properties(rocm_forward_test.cpp
    PROPERTIES COMPILE_OPTIONS "-O2")
  add_test(NAME flint.rocm.forward COMMAND flint_rocm_forward_test)
  set_tests_properties(flint.rocm.forward PROPERTIES
    LABELS "flint;gpu;external-model;slow" SKIP_RETURN_CODE 77)
endif()
