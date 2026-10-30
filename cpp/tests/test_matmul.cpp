#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "runtime/matmul.h"

// Weights use realistic magnitudes (row scale ~ 0.02 / 127) so the sums stay small.
TEST(Matmul, Avx2DotMatchesScalarWithin1e4) {
  if (!cpu_has_avx2()) GTEST_SKIP() << "AVX2 not available";
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
  std::uniform_int_distribution<int> wd(-127, 127);
  for (int n : {8, 16, 27, 64, 128, 512}) {
    for (int trial = 0; trial < 50; ++trial) {
      std::vector<float> x(n);
      std::vector<int8_t> w(n);
      for (auto& v : x) v = xd(rng);
      for (auto& v : w) v = static_cast<int8_t>(wd(rng));
      float s = dot_scalar(x.data(), w.data(), n) * (0.02f / 127.0f);
      float a = dot_avx2(x.data(), w.data(), n) * (0.02f / 127.0f);
      EXPECT_LE(std::fabs(s - a), 1e-4f) << "n=" << n;
    }
  }
}

TEST(Matmul, LinearScalarAndAvx2Agree) {
  if (!cpu_has_avx2()) GTEST_SKIP() << "AVX2 not available";
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
  std::uniform_int_distribution<int> wd(-127, 127);
  QLinear w;
  w.out = 96;
  w.in = 128;
  w.scale.assign(w.out, 0.02f / 127.0f);
  w.q.resize(w.out * w.in);
  for (auto& v : w.q) v = static_cast<int8_t>(wd(rng));
  const int T = 5;
  std::vector<float> X(T * w.in), Ys(T * w.out), Ya(T * w.out);
  for (auto& v : X) v = xd(rng);
  linear(w, X.data(), T, Ys.data(), false);
  linear(w, X.data(), T, Ya.data(), true);
  for (size_t i = 0; i < Ys.size(); ++i) EXPECT_LE(std::fabs(Ys[i] - Ya[i]), 1e-4f);
}
