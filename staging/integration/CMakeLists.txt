# Serve-seam glue between the shared HTTP backend and the staging model
# engines: fork architecture dispatch with no presence in src/. Referenced
# by gufo_serving; its engine and serving symbols resolve in the consumer's
# final link, so this target does not link gufo_serving back.
add_library(gufo_staging_integration STATIC
  fork_backend.cpp)
target_include_directories(gufo_staging_integration
  PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_staging_integration
  PUBLIC gufo_core gufo_ext_runtime)
