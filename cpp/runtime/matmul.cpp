#include "runtime/matmul.h"

#include <immintrin.h>

float dot_scalar(const float* x, const int8_t* w, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += x[i] * static_cast<float>(w[i]);
  return acc;
}

__attribute__((target("avx2,fma"))) float dot_avx2(const float* x, const int8_t* w, int n) {
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + i));
    __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), wf, acc);
  }
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float r = _mm_cvtss_f32(s);
  for (; i < n; ++i) r += x[i] * static_cast<float>(w[i]);
  return r;
}

bool cpu_has_avx2() { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }

void linear(const QLinear& w, const float* X, int T, float* Y, bool avx2) {
  for (int t = 0; t < T; ++t) {
    const float* x = X + static_cast<size_t>(t) * w.in;
    float* y = Y + static_cast<size_t>(t) * w.out;
    for (int r = 0; r < w.out; ++r) {
      const int8_t* row = w.q.data() + static_cast<size_t>(r) * w.in;
      float d = avx2 ? dot_avx2(x, row, w.in) : dot_scalar(x, row, w.in);
      y[r] = d * w.scale[r];
    }
  }
}
