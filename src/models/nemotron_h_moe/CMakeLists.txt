# NVIDIA Nemotron 3.5 Bolt (nemotron_h_moe) model package. A hybrid trunk of
# Mamba-2 blocks, NoPE grouped-query attention blocks and latent-MoE FFN
# blocks with heterogeneous per-layer expert counts, plus the optional
# nemotron_omni CLIP vision projector (deferred). Model-private kernels only:
# nothing here modifies another model tree (see docs/models/
# nvidia-nemotron-bolt-74B-A9B/STATUS.md golden rules).
add_library(gufo_nemotron_h_moe STATIC
  config.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_nemotron_h_moe PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_nemotron_h_moe PUBLIC gufo_core Threads::Threads)

# The scalar oracle is built only by explicit validation targets, never by
# production or routine checks.
add_library(gufo_nemotron_h_moe_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_nemotron_h_moe_reference PUBLIC
  gufo_nemotron_h_moe)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(NB_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  set(NB_MMQ_DIR "${NB_ROCM_DIR}/mmq")

  # HIP quantized matrix kernels adapted from llama.cpp (vendored per
  # kernels/rocm/mmq/VENDOR.md), isolated in the nmb_mmq namespace. IQ2_XXS
  # and NVFP4 instantiations back the Bolt experts and Mamba-2 projections.
  set(NB_MMQ_SOURCES
    ${NB_MMQ_DIR}/nmb_ggml_stubs.hip.cpp
    ${NB_MMQ_DIR}/nmb_mmq.hip.cpp
    ${NB_MMQ_DIR}/mmid.hip.cpp
    ${NB_MMQ_DIR}/mmvq.hip.cpp
    ${NB_MMQ_DIR}/moe_wave64.hip.cpp
    ${NB_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_nemotron_h_moe_mmq STATIC ${NB_MMQ_SOURCES})
  set_source_files_properties(${NB_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${NB_MMQ_DIR}/moe_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  set_target_properties(gufo_nemotron_h_moe_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_nemotron_h_moe_mmq PRIVATE
    GGML_HIP_NO_VMM ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_nemotron_h_moe_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_nemotron_h_moe_mmq SYSTEM PUBLIC
    ${PROJECT_SOURCE_DIR} ${NB_MMQ_DIR})
  target_link_libraries(gufo_nemotron_h_moe_mmq PUBLIC hip::host roc::hipblas)

  # Fused elementwise kernels and the Mamba-2 scan backing the GPU executor.
  add_library(gufo_nemotron_h_moe_kernels STATIC
    ${NB_ROCM_DIR}/nb_ops.hip.cpp
    ${NB_ROCM_DIR}/nb_mamba2.hip.cpp
    ${NB_ROCM_DIR}/nb_attention.hip.cpp)
  set_source_files_properties(${NB_ROCM_DIR}/nb_ops.hip.cpp
                              ${NB_ROCM_DIR}/nb_mamba2.hip.cpp
                              ${NB_ROCM_DIR}/nb_attention.hip.cpp
    PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_nemotron_h_moe_kernels PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_nemotron_h_moe_kernels SYSTEM PUBLIC
    ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_nemotron_h_moe_kernels PUBLIC hip::host)
endif()
