// Parity probe against the pinned llama.cpp reference: feeds raw token ids
// one per decode step at positions 0..N-1 and dumps raw f32 logits after
// each step. dump path
// `tests/models/nemotron_h_moe/reference_test.cpp` (GUFO_NEMOTRON_LOGITS_DUMP)
//
// Build (static libs from the pinned checkout):
//   clang++ -std=c++17 -O2 -I<llama.cpp>/include nemotron_logits_probe.cpp \
//     <build>/src/libllama.a <build>/ggml/src/libggml*.a -lm -lstdc++ -o probe
#include <llama-ext.h>
#include <llama.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool ParseTokens(const std::string& csv, std::vector<llama_token>* out) {
  size_t pos = 0;
  while (pos < csv.size()) {
    size_t comma = csv.find(',', pos);
    if (comma == std::string::npos) {
      comma = csv.size();
    }
    char* end = nullptr;
    long v = std::strtol(csv.substr(pos, comma - pos).c_str(), &end, 10);
    if (end == csv.substr(pos, comma - pos).c_str() || v < 0) {
      return false;
    }
    out->push_back(static_cast<llama_token>(v));
    pos = comma + 1;
  }
  return !out->empty();
}

uint32_t ArgMax(const float* v, int n) {
  uint32_t best = 0;
  for (int i = 1; i < n; ++i) {
    if (v[i] > v[best]) {
      best = static_cast<uint32_t>(i);
    }
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path = argc > 1 ? argv[1] : "";
  std::string dump_path = argc > 2 ? argv[2] : "";
  std::vector<llama_token> tokens;
  if (argc > 3 && !ParseTokens(argv[3], &tokens)) {
    std::fprintf(stderr, "bad tokens list\n");
    return 1;
  }
  if (tokens.empty()) {
    tokens = {100, 200, 300};
  }
  if (model_path.empty() || dump_path.empty()) {
    std::fprintf(stderr, "usage: %s MODEL.GGUF DUMP.F32 [tok,tok,...]\n",
                 argv[0]);
    return 1;
  }

  llama_backend_init();
  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0;
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
  if (model == nullptr) {
    std::fprintf(stderr, "model load failed\n");
    return 1;
  }
  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 64;
  cparams.n_batch = 8;
  llama_context* ctx = llama_init_from_model(model, cparams);
  if (ctx == nullptr) {
    std::fprintf(stderr, "context init failed\n");
    return 1;
  }
  const int n_out = llama_model_n_embd_out(model);
  const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  const int n_layer = llama_model_n_layer(model);
  std::fprintf(stderr, "n_vocab=%d n_out=%d n_layer=%d n_ctx=%u\n", n_vocab,
               n_out, n_layer, cparams.n_ctx);

  std::ofstream dump(dump_path, std::ios::binary | std::ios::trunc);
  if (!dump) {
    std::fprintf(stderr, "cannot open %s\n", dump_path.c_str());
    return 1;
  }
  const char* layer_path = std::getenv("GUFO_LLAMA_LAYER_DUMP");
  std::ofstream layer_dump;
  if (layer_path != nullptr && layer_path[0] != '\0') {
    layer_dump.open(layer_path, std::ios::binary | std::ios::trunc);
    if (!layer_dump) {
      std::fprintf(stderr, "cannot open %s\n", layer_path);
      return 1;
    }
    for (int lid = 0; lid < n_layer; ++lid) {
      llama_set_embeddings_layer_inp(ctx, lid, true);
    }
  }
  llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
  llama_seq_id seq = 0;
  for (size_t i = 0; i < tokens.size(); ++i) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = tokens[i];
    batch.pos[0] = static_cast<llama_pos>(i);
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = seq;
    batch.logits[0] = 1;
    if (llama_decode(ctx, batch) != 0) {
      std::fprintf(stderr, "decode failed at step %zu\n", i);
      return 1;
    }
    const float* logits = llama_get_logits_ith(ctx, 0);
    if (logits == nullptr) {
      std::fprintf(stderr, "no logits at step %zu\n", i);
      return 1;
    }
    double sum = 0.0;
    bool finite = true;
    for (int v = 0; v < n_vocab; ++v) {
      sum += static_cast<double>(logits[v]);
      finite = finite && std::isfinite(logits[v]);
    }
    std::printf("llama step %zu argmax %u sum %.6g%s\n", i,
                ArgMax(logits, n_vocab), sum, finite ? "" : " NONFINITE");
    dump.write(reinterpret_cast<const char*>(logits),
               static_cast<std::streamsize>(n_vocab) * sizeof(float));
    if (layer_dump) {
      for (int lid = 0; lid < n_layer; ++lid) {
        float* embd = llama_get_embeddings_layer_inp(ctx, lid);
        if (embd == nullptr) {
          std::fprintf(stderr, "no layer input at step %zu layer %d\n", i, lid);
          return 1;
        }
        layer_dump.write(reinterpret_cast<const char*>(embd),
                         static_cast<std::streamsize>(n_out) * sizeof(float));
      }
    }
    llama_batch_free(batch);
  }
  dump.close();
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
