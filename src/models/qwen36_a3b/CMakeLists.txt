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