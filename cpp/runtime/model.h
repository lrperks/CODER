#pragma once
#include <string>
#include <vector>

#include "runtime/matmul.h"

// Char-level transformer loaded from the int8 export written by python/policy/export.py.
struct Layer {
  std::vector<float> ln1_g, ln1_b, ln2_g, ln2_b;
  QLinear qkv, proj, fc1, fc2;
};

struct Cache {  // KV cache: [layer][pos][d]
  int len = 0;
  std::vector<std::vector<float>> k, v;
};

class Model {
 public:
  explicit Model(const std::string& path);

  int vocab() const { return vocab_; }
  int max_len() const { return max_len_; }
  Cache new_cache() const;
  // Appends T tokens to the cache and writes the logits of the last one to logits[vocab].
  void forward(Cache& cache, const std::vector<int>& toks, float* logits, bool avx2) const;

 private:
  int vocab_ = 0, max_len_ = 0, d_ = 0, heads_ = 0, ff_ = 0;
  int layers_n_ = 0;
  std::vector<float> tok_emb_, pos_emb_, lnf_g_, lnf_b_;
  std::vector<Layer> layers_;
  QLinear head_;
};
