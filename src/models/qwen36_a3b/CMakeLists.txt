# Qwen3.6-35B-A3B (qwen35moe) model package. Isolated from the Qwen3.8-27B
# (qwen35) and Qwen3.8-Flash-Next (qwen4exp) engines: it carries no
# hyper-connections, no QSA indexer and no PLE n-gram embedding. Only the
# metadata loader is built until the runtime lands.
add_library(gufo_qwen36_a3b STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_qwen36_a3b PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen36_a3b PUBLIC gufo_core Threads::Threads)

# The scalar oracle is built only by explicit validation targets, never by
# production or routine checks.
add_library(gufo_qwen36_a3b_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_qwen36_a3b_reference PUBLIC
  gufo_qwen36_a3b)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  # Fused operators outside the quantized GEMM tier. Built only by explicit
  # GPU validation targets until the executor consumes them. -ffast-math is
  # deliberately omitted so expf/log1pf stay faithful to the scalar oracle.
  add_library(gufo_qwen36_a3b_kernels STATIC EXCLUDE_FROM_ALL
    kernels/rocm/kernels.hip.cpp
    kernels/rocm/gemv.hip.cpp)
  set_source_files_properties(kernels/rocm/kernels.hip.cpp
    kernels/rocm/gemv.hip.cpp PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_qwen36_a3b_kernels PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_qwen36_a3b_kernels PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_qwen36_a3b_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_qwen36_a3b_kernels PUBLIC hip::host)

  # Host-side driver: streams the artifact to the GPU and runs the trunk and
  # MTP draft one token at a time. Built only by explicit GPU validation
  # targets until the engine and CLI consume it.
  add_library(gufo_qwen36_a3b_executor STATIC EXCLUDE_FROM_ALL
    kernels/rocm/device_model.cpp
    kernels/rocm/executor.cpp)
  set_target_properties(gufo_qwen36_a3b_executor PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_qwen36_a3b_executor PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_qwen36_a3b_executor PUBLIC
    gufo_qwen36_a3b_kernels gufo_qwen36_a3b hip::host)

  # Gufo-owned API over the ROCm runtime: one resident model and one session.
  # Built only by explicit GPU validation targets until the CLI consumes it.
  add_library(gufo_qwen36_a3b_engine STATIC EXCLUDE_FROM_ALL
    engine.cpp)
  set_target_properties(gufo_qwen36_a3b_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_qwen36_a3b_engine PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_qwen36_a3b_engine PUBLIC
    gufo_qwen36_a3b_executor gufo_qwen36_a3b gufo_core)
endif()