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


MODEL_Q4="/home/praburaja/projects/llm/models/gguf/Qwen3.6-35B-A3B-UD_Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_Q8_0="/home/praburaja/projects/llm/models/gguf/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-Q8_0.gguf"
MODEL_Q8="/home/praburaja/projects/llm/models/gguf/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
MMPROJ="/home/praburaja/projects/llm/models/gguf/Qwen3.6-35B-A3B-MTP/mmproj-BF16.gguf"

build/gpu-test/gufo serve llm -v -i "0.0.0.0" -p 8083 -c 131072 -n 32768 --model "$MODEL_Q8_0" --sessions 1 --attn-window 2048 --attn-sink 4 --speculative mtp --min-draft-tokens 2 --draft-tokens 2
#build/gpu-test/gufo chat -v -n 32768 --model "$MODEL_Q8"
#build/gpu-test/gufo prompt -v -n 2048 --model "$MODEL_Q8" -p "Capital of Russia is"
