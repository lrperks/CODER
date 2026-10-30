#pragma once
#include <cstdint>
#include <vector>

// int8 weight matrix with one fp32 scale per output row.
struct QLinear {
  int out = 0, in = 0;
  std::vector<float> scale;
  std::vector<int8_t> q;  // row-major [out][in]
};

// Dot product of fp32 x with int8 w, n elements. Scalar reference and AVX2 paths.
float dot_scalar(const float* x, const int8_t* w, int n);
float dot_avx2(const float* x, const int8_t* w, int n);
bool cpu_has_avx2();

// Y[t][r] = scale[r] * dot(X[t], q[r]) for t < T.
void linear(const QLinear& w, const float* X, int T, float* Y, bool avx2);
