# Gemma-4-26B-A4B (gemma4 / gemma4-assistant / gemma4v) model package.
# Isolated from the other model engines. Only the metadata loader is built
# until the runtime lands.
add_library(gufo_gemma4 STATIC
  chat_template.cpp
  config.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_gemma4 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_gemma4 PUBLIC gufo_core Threads::Threads)

# The scalar oracle is built only by explicit validation targets, never by
# production or routine checks.
add_library(gufo_gemma4_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_gemma4_reference PUBLIC gufo_gemma4)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  # Quantized GEMV/GEMM tier, the routed F16 WMMA MoE and the fused routing
  # operators. Built only by explicit GPU validation targets until the
  # executor consumes them. -ffast-math is deliberately omitted so expf/tanhf
  # stay faithful to the scalar oracle.
  add_library(gufo_gemma4_kernels STATIC EXCLUDE_FROM_ALL
    kernels/rocm/attention.hip.cpp
    kernels/rocm/fused.hip.cpp
    kernels/rocm/gemv.hip.cpp
    kernels/rocm/gemm.hip.cpp
    kernels/rocm/routed_f16.hip.cpp)
  set_source_files_properties(kernels/rocm/attention.hip.cpp
    kernels/rocm/fused.hip.cpp
    kernels/rocm/gemv.hip.cpp kernels/rocm/gemm.hip.cpp
    kernels/rocm/routed_f16.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_gemma4_kernels PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_gemma4_kernels PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_gemma4_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  # The batched GEMM tier reuses the Qwen3.8-Flash-Next mmq routed/dense path
  # (Q8_0) and hipBLAS (F32/BF16); the mmq target exports its header directory
  # and roc::hipblas transitively, so weights stay in their native encoding.
  target_link_libraries(gufo_gemma4_kernels PUBLIC
    gufo_qwen38_flash_next_mmq hip::host)

  # Host-side driver: streams the artifact to the GPU and runs the trunk one
  # token at a time. Built only by explicit GPU validation targets until the
  # engine and CLI consume it.
  add_library(gufo_gemma4_executor STATIC EXCLUDE_FROM_ALL
    kernels/rocm/device_model.cpp
    kernels/rocm/executor.cpp)
  set_target_properties(gufo_gemma4_executor PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_gemma4_executor PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_gemma4_executor PUBLIC
    gufo_gemma4_kernels gufo_gemma4 hip::host)

  # CLI-facing engine: the Model/Session API over the executor and tokenizer.
  # Host code (hipMemcpy only), so it compiles as CXX and links the executor's
  # hip::host transitively.
  add_library(gufo_gemma4_engine STATIC EXCLUDE_FROM_ALL
    engine.cpp)
  set_target_properties(gufo_gemma4_engine PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_gemma4_engine PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_gemma4_engine PUBLIC
    gufo_gemma4_executor gufo_gemma4 gufo_core)
endif()