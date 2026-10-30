#include "runtime/decode.h"

#include <cmath>
#include <random>
#include <vector>

#include "runtime/grammar.h"

namespace {

// Picks the next token among `allowed` by argmax (temperature 0) or sampling.
int choose(const std::vector<float>& logits, const std::bitset<128>& allowed, float temperature,
           std::mt19937_64& rng) {
  int best = -1;
  if (temperature <= 0.0f) {
    for (int c = 0; c < 128; ++c) {
      if (allowed.test(c) && (best < 0 || logits[c] > logits[best])) best = c;
    }
    return best;
  }
  float mx = -INFINITY;
  for (int c = 0; c < 128; ++c) {
    if (allowed.test(c) && logits[c] > mx) mx = logits[c];
  }
  std::vector<double> p(128, 0.0);
  double sum = 0.0;
  for (int c = 0; c < 128; ++c) {
    if (allowed.test(c)) {
      p[c] = std::exp((logits[c] - mx) / temperature);
      sum += p[c];
    }
  }
  std::uniform_real_distribution<double> u(0.0, sum);
  double r = u(rng);
  for (int c = 0; c < 128; ++c) {
    if (!allowed.test(c)) continue;
    best = c;
    r -= p[c];
    if (r <= 0.0) break;
  }
  return best;
}

}  // namespace

Decoded decode(const Model& model, const std::string& prompt, const DecodeOpts& opts) {
  const Nfa& nfa = program_nfa();
  Nfa::States g = nfa.start();
  std::mt19937_64 rng(opts.seed);
  Cache cache = model.new_cache();
  std::vector<int> pending;  // tokens not yet run through the model
  for (unsigned char c : prompt) pending.push_back(c);
  std::vector<float> logits(model.vocab());
  Decoded out;
  while (out.tokens < opts.max_new) {
    std::bitset<128> allowed;
    if (opts.mode == Mode::Edit) {
      allowed = nfa.allowed(g);
      if (allowed.count() == 0) break;
      if (allowed.count() == 1 && allowed.test(0)) break;  // complete program
      if (allowed.count() == 1) {  // forced: no sampling, defer the model pass
        int c = 0;
        while (!allowed.test(c)) ++c;
        pending.push_back(c);
        g = nfa.step(g, static_cast<char>(c));
        out.text += static_cast<char>(c);
        ++out.tokens;
        ++out.forced;
        continue;
      }
    } else {
      allowed.set();
    }
    model.forward(cache, pending, logits.data(), opts.avx2);
    pending.clear();
    int c = choose(logits, allowed, opts.temperature, rng);
    if (c == 0 && opts.stop_at_eos) break;
    pending.push_back(c);
    out.text += static_cast<char>(c);
    ++out.tokens;
    if (opts.mode == Mode::Edit) g = nfa.step(g, static_cast<char>(c));
  }
  return out;
}
