# Ling 3.0 Flash (bailingmoe3) model package. A hybrid trunk of Kimi Delta
# Attention (KDA) linear layers and absorbed MLA layers, group-limited MoE
# routing with one shared expert, and a self-speculative MTP block. The CLI
# and serve runners link the engine target below.
add_library(gufo_ling3_flash STATIC
  chat_template.cpp
  config.cpp
  tool_syntax.cpp
  weights.cpp)

target_include_directories(gufo_ling3_flash PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_ling3_flash PUBLIC gufo_core Threads::Threads)

# The scalar oracle is built only by explicit validation targets, never by
# production or routine checks.
add_library(gufo_ling3_flash_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_ling3_flash_reference PUBLIC
  gufo_ling3_flash)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  # Quantized GEMV tier (decodes Q4_K/Q5_K/Q6_K/Q8_0 trunk rows and the Q4_0
  # MTP rows on the fly), the fused single-token operators between the
  # projections, and the split-KV absorbed-MLA decode attention. Built only by
  # explicit GPU validation targets until the engine and CLI consume them.
  # -ffast-math is deliberately omitted so expf/sigmoid stay faithful to the
  # scalar oracle.
  add_library(gufo_ling3_flash_kernels STATIC EXCLUDE_FROM_ALL
    kernels/rocm/gemv.hip.cpp
    kernels/rocm/kernels.hip.cpp
    kernels/rocm/routed_f16.hip.cpp
    kernels/rocm/attention.hip.cpp)
  set_source_files_properties(kernels/rocm/gemv.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/kernels.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/routed_f16.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_source_files_properties(kernels/rocm/attention.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_ling3_flash_kernels PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_ling3_flash_kernels PUBLIC
    ${PROJECT_SOURCE_DIR})
  # The fused MLA prefill kernel spills 316 B/lane to scratch (DRAM) when
  # NDEBUG is active: the allocator schedules a pathological region once the
  # host-side assert paths vanish, and the kernel runs ~6x slower streaming
  # register spills. -UNDEBUG restores spill-free codegen; keep it on the
  # whole kernels target so no future kernel inherits the same artifact.
  target_compile_options(gufo_ling3_flash_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-UNDEBUG;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_ling3_flash_kernels PUBLIC hip::host)

  # Host-side driver: streams the artifact to the GPU and runs the trunk and
  # MTP draft one token at a time. Built only by explicit GPU validation
  # targets until the engine and CLI consume it.
  add_library(gufo_ling3_flash_executor STATIC EXCLUDE_FROM_ALL
    kernels/rocm/device_model.cpp
    kernels/rocm/executor.cpp)
  set_target_properties(gufo_ling3_flash_executor PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_ling3_flash_executor PUBLIC
    ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_ling3_flash_executor PUBLIC
    gufo_ling3_flash_kernels gufo_ling3_flash gufo_core hip::host)

  # Gufo-owned API over the ROCm runtime: one resident model and independent
  # sessions. Linked by the CLI and the serve runner.
  add_library(gufo_ling3_flash_engine STATIC EXCLUDE_FROM_ALL
    engine.cpp)
  set_target_properties(gufo_ling3_flash_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_ling3_flash_engine PUBLIC
    ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_ling3_flash_engine PUBLIC
    gufo_ling3_flash_executor gufo_ling3_flash gufo_core)
endif()