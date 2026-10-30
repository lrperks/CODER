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

// Four rows share each load of x and keep four independent accumulators.
__attribute__((target("avx2,fma"))) static inline __m256 load_w8(const int8_t* w) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w))));
}

__attribute__((target("avx2,fma"))) static float hsum8(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

__attribute__((target("avx2,fma"))) static void dot4_avx2(const float* x, const int8_t* w, int n, float* out) {
  const int8_t *w1 = w + n, *w2 = w1 + n, *w3 = w2 + n;
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 xv = _mm256_loadu_ps(x + i);
    a0 = _mm256_fmadd_ps(xv, load_w8(w + i), a0);
    a1 = _mm256_fmadd_ps(xv, load_w8(w1 + i), a1);
    a2 = _mm256_fmadd_ps(xv, load_w8(w2 + i), a2);
    a3 = _mm256_fmadd_ps(xv, load_w8(w3 + i), a3);
  }
  out[0] = hsum8(a0);
  out[1] = hsum8(a1);
  out[2] = hsum8(a2);
  out[3] = hsum8(a3);
  for (; i < n; ++i) {
    const float xi = x[i];
    out[0] += xi * w[i];
    out[1] += xi * w1[i];
    out[2] += xi * w2[i];
    out[3] += xi * w3[i];
  }
}

void linear(const QLinear& w, const float* X, int T, float* Y, bool avx2) {
  for (int t = 0; t < T; ++t) {
    const float* x = X + static_cast<size_t>(t) * w.in;
    float* y = Y + static_cast<size_t>(t) * w.out;
    int r = 0;
    if (avx2) {
      for (; r + 4 <= w.out; r += 4) {
        float d[4];
        dot4_avx2(x, w.q.data() + static_cast<size_t>(r) * w.in, w.in, d);
        for (int j = 0; j < 4; ++j) y[r + j] = d[j] * w.scale[r + j];
      }
    }
    for (; r < w.out; ++r) {
      const int8_t* row = w.q.data() + static_cast<size_t>(r) * w.in;
      float d = avx2 ? dot_avx2(x, row, w.in) : dot_scalar(x, row, w.in);
      y[r] = d * w.scale[r];
    }
  }
}
