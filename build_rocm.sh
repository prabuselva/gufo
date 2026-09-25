#!/bin/bash


export HIP_PATH=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export HIP_CLANG_PATH=/opt/rocm/llvm/bin
export HIP_INCLUDE_PATH=/opt/rocm/include
export HIP_LIB_PATH=/opt/rocm/lib
export HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:${LIBRARY_PATH:-}
export CPATH=/opt/rocm/include:${CPATH:-}
export PKG_CONFIG_PATH=/opt/rocm/lib/pkgconfig:${PKG_CONFIG_PATH:-}


# Configure the project
# Build the project (adjust -j value based on your CPU cores)
cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
#cmake --preset hardware-test
#cmake --build --preset hardware-test --parallel 4
