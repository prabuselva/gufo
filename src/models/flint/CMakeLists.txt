# Flint Preview 40B-A2B (flint) model package: 49-layer trunk with 2 leading
# dense FFN layers, 256-expert sigmoid top-6 MoE plus 2 shared experts, and
# gated per-layer rotary/window attention with QK norms and attention sinks.
# Only the metadata loader and weight binder are built until the runtime lands.
add_library(gufo_flint STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_flint PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_flint PUBLIC gufo_core Threads::Threads)

# The scalar oracle is built only by explicit validation targets, never by
# production or routine checks.
add_library(gufo_flint_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_flint_reference PUBLIC gufo_flint)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")
