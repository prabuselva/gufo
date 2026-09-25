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


MODEL="/home/praburaja/projects/llm/models/gguf/Qwen3.8-Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
MTP="/home/praburaja/projects/llm/models/gguf/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-Q8_0_unsloth.gguf"
build/release/gufo serve llm -i "0.0.0.0" -p 8083 -c 131072 -n 32768 --model "$MODEL"

