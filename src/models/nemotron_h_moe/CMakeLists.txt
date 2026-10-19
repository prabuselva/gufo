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
