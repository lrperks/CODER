#pragma once
#include <cstdint>
#include <string>

#include "runtime/model.h"

enum class Mode {
  Edit,  // grammar mask over the DSL; forced tokens skip sampling
  Free,  // no mask, no forced tokens (baseline)
};

struct DecodeOpts {
  Mode mode = Mode::Edit;
  int max_new = 160;
  float temperature = 0.0f;  // 0 = greedy
  uint64_t seed = 0;
  bool stop_at_eos = true;
  bool avx2 = true;
};

struct Decoded {
  std::string text;
  int tokens = 0;  // emitted characters, excluding EOS
  int forced = 0;  // of which emitted by the forced-token shortcut
};

Decoded decode(const Model& model, const std::string& prompt, const DecodeOpts& opts);
