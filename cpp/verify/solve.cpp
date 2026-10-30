#include "verify/solve.h"

#include <atomic>
#include <chrono>
#include <thread>

#include "edits/edits.h"
#include "edits/program.h"

namespace {

// Decodes candidate k for one sample, verifies it, and updates stats.
// Returns true when the candidate passes.
bool try_candidate(const Model& model, const Sample& s, const SolveOpts& o, uint64_t sample_idx, int k,
                   SampleStats& st) {
  DecodeOpts d;
  d.mode = o.mode;
  d.max_new = o.max_new;
  d.temperature = k == 0 ? 0.0f : 1.0f;
  d.seed = o.seed * 1000003u + sample_idx * 16u + static_cast<uint64_t>(k);
  d.avx2 = o.avx2;
  Decoded dec = decode(model, s.prompt, d);
  st.tokens += dec.tokens;
  st.forced += dec.forced;
  std::string candidate;
  if (o.mode == Mode::Edit) {
    auto prog = parse_program(dec.text);
    auto edited = prog ? apply_program(s.src, *prog) : std::nullopt;
    if (!edited) return false;
    candidate = *edited;
  } else {
    candidate = dec.text;
  }
  st.verify_calls++;
  return run_test(o.verify, candidate, s.test);
}

}  // namespace

std::vector<SampleStats> solve(const Model& model, const std::vector<Sample>& samples, const SolveOpts& o) {
  std::vector<SampleStats> out(samples.size());
  std::atomic<size_t> next{0};
  std::vector<std::thread> pool;
  for (int w = 0; w < o.workers; ++w) {
    pool.emplace_back([&] {
      for (size_t i; (i = next.fetch_add(1)) < samples.size();) {
        SampleStats& st = out[i];
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < o.k; ++k) {
          if (try_candidate(model, samples[i], o, i, k, st)) {
            st.pass4 = true;
            st.pass1 = k == 0;
            break;
          }
        }
        st.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      }
    });
  }
  for (auto& t : pool) t.join();
  return out;
}
