# Metadata checks are CPU-only and run in the routine suite. Operator and
# model checks are added as their kernels land.
add_custom_target(flint_tests)

add_executable(flint_config_test config_test.cpp)
target_link_libraries(flint_config_test PRIVATE gufo_flint)
add_test(NAME flint.config COMMAND flint_config_test)
set_tests_properties(flint.config PROPERTIES LABELS "flint;cpu")
add_dependencies(flint_tests flint_config_test)

# Weight-table binding runs against the real artifact when GUFO_FLINT_GGUF
# names it and skips (77) otherwise.
add_executable(flint_weights_test weights_test.cpp)
target_link_libraries(flint_weights_test PRIVATE gufo_flint)
add_test(NAME flint.weights COMMAND flint_weights_test)
set_tests_properties(flint.weights PROPERTIES
  LABELS "flint;cpu;external-model" SKIP_RETURN_CODE 77)
add_dependencies(flint_tests flint_weights_test)
