source rocm_env.sh
cmake --build --preset gpu-test --target gufo --parallel 4

MODEL_Q8="/home/praburaja/projects/llm/models/gguf/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
#MODEL_Q8="/home/praburaja/projects/llm/models/gguf/Qwen3.6-35B-A3B-MTP/Qwen3.6-35B-A3B-Q8_0.gguf"

GUFO_QWEN36_PROFILE=1 ./build/gpu-test/gufo serve llm -m $MODEL_Q8 -c 16384 --speculative mtp --min-draft-tokens 3 --draft-tokens 3
#--min-draft-tokens 2 --draft-tokens 6
