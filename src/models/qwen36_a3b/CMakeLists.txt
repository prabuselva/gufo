# Qwen3.6-35B-A3B (qwen35moe) model package. Isolated from the Qwen3.8-27B
# (qwen35) and Qwen3.8-Flash-Next (qwen4exp) engines: it carries no
# hyper-connections, no QSA indexer and no PLE n-gram embedding. Only the
# metadata loader is built until the runtime lands.
add_library(gufo_qwen36_a3b STATIC
  config.cpp)

target_include_directories(gufo_qwen36_a3b PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen36_a3b PUBLIC gufo_core Threads::Threads)