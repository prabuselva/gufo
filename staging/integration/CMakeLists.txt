# Serve- and CLI-seam glue between the shared drivers and the staging model
# engines: staging architecture dispatch with no presence in src/. Referenced
# by gufo_serving and the gufo CLI; its engine and serving symbols resolve in
# the consumer's final link, so this target does not link them back.
add_library(gufo_staging_integration STATIC
  staging_backend.cpp
  staging_cli.cpp)
target_include_directories(gufo_staging_integration
  PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_staging_integration
  PUBLIC gufo_core gufo_ext_runtime)
