# Shared dense binary16 prefill GEMM: Q8_0 weights dequantized to F16 in
# LDS over F16 activations, with a tile-plan hook the models select per
# projection (staging/core/gemm/dense_f16_gemm.hpp).
add_library(gufo_staging_gemm STATIC EXCLUDE_FROM_ALL
  dense_f16_gemm.hip.cpp)
set_source_files_properties(dense_f16_gemm.hip.cpp
  PROPERTIES LANGUAGE HIP)
set_target_properties(gufo_staging_gemm PROPERTIES
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(gufo_staging_gemm PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(gufo_staging_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-fno-finite-math-only;-Wno-unused-command-line-argument>)
target_link_libraries(gufo_staging_gemm PUBLIC hip::host)
