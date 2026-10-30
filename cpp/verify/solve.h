#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "runtime/decode.h"
#include "verify/verify.h"

struct Sample {
  std::string prompt;  // policy input (see render_prompt)
  std::string src;     // buggy source
  std::string test;    // failing test
};

struct SampleStats {
  bool pass1 = false;  // the greedy candidate passes
  bool pass4 = false;  // some candidate among the first K passes
  int tokens = 0;      // characters emitted, all candidates tried
  int forced = 0;      // of which from the forced-token shortcut
  int verify_calls = 0;
  double wall_s = 0.0;  // decode and verify time for this sample
};

struct SolveOpts {
  Mode mode = Mode::Edit;
  int k = 4;
  int max_new = 160;
  int workers = 4;
  uint64_t seed = 0;
  bool avx2 = true;
  VerifyOpts verify;
};

// Candidate 0 is greedy; candidates 1..K-1 are sampled at temperature 1. Stops at the
// first candidate that passes. Edit programs that do not apply are not verified.
std::vector<SampleStats> solve(const Model& model, const std::vector<Sample>& samples, const SolveOpts& opts);
