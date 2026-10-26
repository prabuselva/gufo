# Metadata checks are CPU-only and run in the routine suite. Operator and
# model checks are added as their kernels land.
add_custom_target(flint_tests)

add_executable(flint_config_test config_test.cpp)
target_link_libraries(flint_config_test PRIVATE gufo_flint)
add_test(NAME flint.config COMMAND flint_config_test)
set_tests_properties(flint.config PROPERTIES LABELS "flint;cpu")
add_dependencies(flint_tests flint_config_test)
