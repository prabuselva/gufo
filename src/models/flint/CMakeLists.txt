# Flint Preview 40B-A2B (flint) model package: 49-layer trunk with 2 leading
# dense FFN layers, 256-expert sigmoid top-6 MoE plus 2 shared experts, and
# gated per-layer rotary/window attention with QK norms and attention sinks.
# Only the metadata loader is built until the runtime lands.
add_library(gufo_flint STATIC
  config.cpp)

target_include_directories(gufo_flint PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_flint PUBLIC gufo_core Threads::Threads)
