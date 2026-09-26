# Metadata checks are CPU-only and run in the routine suite. Operator and
# model checks are added as their kernels land.
add_custom_target(qwen36_a3b_tests)

add_executable(qwen36_a3b_config_test config_test.cpp)
target_link_libraries(qwen36_a3b_config_test PRIVATE gufo_qwen36_a3b)
add_test(NAME qwen36_a3b.config COMMAND qwen36_a3b_config_test)
set_tests_properties(qwen36_a3b.config PROPERTIES LABELS "qwen36_a3b;cpu")
add_dependencies(qwen36_a3b_tests qwen36_a3b_config_test)