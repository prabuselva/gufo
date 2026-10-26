# Flint Preview 40B-A2B (flint) model package: 49-layer trunk with 2 leading
# dense FFN layers, 256-expert sigmoid top-6 MoE plus 2 shared experts, and
# gated per-layer rotary/window attention with QK norms and attention sinks.
# The metadata loader, gpt2-BPE tokenizer and weight binder are built here;
# the runtime lands with the engine.
add_library(gufo_flint STATIC
  config.cpp
  tokenizer.cpp
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

if (ENGINE_ENABLE_HIP)
  # Flint-private fused operators and linear tiers. Built only by explicit
  # GPU validation targets until the engine consumes them.
  add_library(gufo_flint_kernels STATIC EXCLUDE_FROM_ALL
    kernels/rocm/kernels.hip.cpp
    kernels/rocm/linear.hip.cpp)
  set_source_files_properties(kernels/rocm/kernels.hip.cpp
    kernels/rocm/linear.hip.cpp PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_flint_kernels PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_flint_kernels PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_flint_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  # The batched Q8_0 tiers reuse the Qwen3.8-Flash-Next mmq path; the target
  # exports its header directory transitively, weights stay native.
  target_link_libraries(gufo_flint_kernels PUBLIC
    gufo_qwen38_flash_next_mmq gufo_flint hip::host)

  # Host-side driver: streams the artifact to the GPU and runs one forward.
  add_library(gufo_flint_executor STATIC EXCLUDE_FROM_ALL
    kernels/rocm/executor.cpp)
  set_target_properties(gufo_flint_executor PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_flint_executor PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_flint_executor PUBLIC
    gufo_flint_kernels gufo_flint hip::host)

  # Model/Session runtime consumed by the CLI runners.
  add_library(gufo_flint_engine STATIC EXCLUDE_FROM_ALL engine.cpp)
  set_target_properties(gufo_flint_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_flint_engine PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_flint_engine PUBLIC gufo_flint_executor)
endif()
