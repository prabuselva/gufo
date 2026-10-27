# Flint Preview 40B-A2B (flint) model package: 49-layer trunk with 2 leading
# dense FFN layers, 256-expert sigmoid top-6 MoE plus 2 shared experts, and
# gated per-layer rotary/window attention with QK norms and attention sinks.
# The metadata loader, gpt2-BPE tokenizer and weight binder are built here;
# the runtime lands with the engine.
add_library(gufo_flint STATIC
  chat_template.cpp
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
  # Flint fork of the vendored mmq tree (see mmq/VENDOR.md): identical at
  # import to the Flash-Next path aside from the flint_ rename, evolved
  # here for the Q8_0 pipeline without touching shared production kernels.
  set(FLINT_MMQ_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/mmq")
  set(FLINT_MMQ_SOURCES
    ${FLINT_MMQ_DIR}/flint_ggml_stubs.hip.cpp
    ${FLINT_MMQ_DIR}/flint_mmq.hip.cpp
    ${FLINT_MMQ_DIR}/mmid.hip.cpp
    ${FLINT_MMQ_DIR}/mmvq.hip.cpp
    ${FLINT_MMQ_DIR}/moe_wave64.hip.cpp
    ${FLINT_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_flint_mmq STATIC EXCLUDE_FROM_ALL ${FLINT_MMQ_SOURCES})
  set_source_files_properties(${FLINT_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${FLINT_MMQ_DIR}/moe_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  set_target_properties(gufo_flint_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_flint_mmq PRIVATE
    GGML_HIP_NO_VMM ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_flint_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_flint_mmq SYSTEM PUBLIC ${FLINT_MMQ_DIR})
  target_link_libraries(gufo_flint_mmq PUBLIC hip::host roc::hipblas)

  # Flint-private fused operators and linear tiers. Built only by explicit
  # GPU validation targets until the engine consumes them.
  add_library(gufo_flint_kernels STATIC EXCLUDE_FROM_ALL
    kernels/rocm/attention_wmma.hip.cpp
    kernels/rocm/kernels.hip.cpp
    kernels/rocm/linear.hip.cpp)
  set_source_files_properties(kernels/rocm/attention_wmma.hip.cpp
    kernels/rocm/kernels.hip.cpp
    kernels/rocm/linear.hip.cpp PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_flint_kernels PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_flint_kernels PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_flint_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  # The batched Q8_0 tiers use the Flint-owned mmq fork; the target exports
  # its header directory transitively, weights stay native.
  target_link_libraries(gufo_flint_kernels PUBLIC
    gufo_flint_mmq gufo_flint hip::host)

  # Flint fork of the Gemma-4 grouped MoE prefill route (see
  # kernels/rocm/moe/VENDOR.md): Q8_0 128-row bucketed WMMA GEMMs with
  # device-side RoutedCompact, evaluated against the mmq route.
  add_library(gufo_flint_moe STATIC EXCLUDE_FROM_ALL
    kernels/rocm/moe/moe_prefill.hip.cpp
    kernels/rocm/moe/routed_compact.hip.cpp)
  set_source_files_properties(kernels/rocm/moe/moe_prefill.hip.cpp
    kernels/rocm/moe/routed_compact.hip.cpp PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_flint_moe PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_flint_moe PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_flint_moe PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_flint_moe PUBLIC hip::host)

  # Host-side driver: streams the artifact to the GPU and runs one forward.
  add_library(gufo_flint_executor STATIC EXCLUDE_FROM_ALL
    kernels/rocm/executor.cpp)
  set_target_properties(gufo_flint_executor PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_flint_executor PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_flint_executor PUBLIC
    gufo_flint_kernels gufo_flint_moe gufo_flint hip::host)

  # Model/Session runtime and the bench runner consumed by the CLI.
  add_library(gufo_flint_engine STATIC EXCLUDE_FROM_ALL
    engine.cpp
    cli_runner.cpp)
  set_target_properties(gufo_flint_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_flint_engine PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_flint_engine PUBLIC gufo_flint_executor)
endif()
