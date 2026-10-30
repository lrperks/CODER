#include "runtime/model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {

void read_bytes(std::FILE* f, void* dst, size_t n) {
  if (std::fread(dst, 1, n, f) != n) throw std::runtime_error("model file truncated");
}

std::vector<float> read_f32(std::FILE* f, size_t n) {
  std::vector<float> v(n);
  read_bytes(f, v.data(), n * sizeof(float));
  return v;
}

QLinear read_qlinear(std::FILE* f) {
  int32_t dims[2];
  read_bytes(f, dims, sizeof dims);
  QLinear w;
  w.out = dims[0];
  w.in = dims[1];
  w.scale = read_f32(f, w.out);
  w.q.resize(static_cast<size_t>(w.out) * w.in);
  read_bytes(f, w.q.data(), w.q.size());
  return w;
}

void layernorm(const float* x, const std::vector<float>& g, const std::vector<float>& b, int T, int d,
               float* y) {
  for (int t = 0; t < T; ++t) {
    const float* xi = x + static_cast<size_t>(t) * d;
    float* yi = y + static_cast<size_t>(t) * d;
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < d; ++i) mean += xi[i];
    mean /= d;
    for (int i = 0; i < d; ++i) var += (xi[i] - mean) * (xi[i] - mean);
    float inv = 1.0f / std::sqrt(var / d + 1e-5f);
    for (int i = 0; i < d; ++i) yi[i] = (xi[i] - mean) * inv * g[i] + b[i];
  }
}

float dot_f32(const float* a, const float* b, int n) {
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

float gelu(float x) {
  return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

}  // namespace

Model::Model(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[4];
  read_bytes(f, magic, 4);
  if (std::memcmp(magic, "VEDM", 4) != 0) throw std::runtime_error("bad magic");
  int32_t hdr[8];
  read_bytes(f, hdr, sizeof hdr);
  vocab_ = hdr[1];
  max_len_ = hdr[2];
  d_ = hdr[3];
  layers_n_ = hdr[4];
  heads_ = hdr[5];
  ff_ = hdr[6];
  tok_emb_ = read_f32(f, static_cast<size_t>(vocab_) * d_);
  pos_emb_ = read_f32(f, static_cast<size_t>(max_len_) * d_);
  for (int l = 0; l < layers_n_; ++l) {
    Layer L;
    L.ln1_g = read_f32(f, d_);
    L.ln1_b = read_f32(f, d_);
    L.qkv = read_qlinear(f);
    L.proj = read_qlinear(f);
    L.ln2_g = read_f32(f, d_);
    L.ln2_b = read_f32(f, d_);
    L.fc1 = read_qlinear(f);
    L.fc2 = read_qlinear(f);
    layers_.push_back(std::move(L));
  }
  lnf_g_ = read_f32(f, d_);
  lnf_b_ = read_f32(f, d_);
  head_ = read_qlinear(f);
  std::fclose(f);
}

Cache Model::new_cache() const {
  Cache c;
  c.k.assign(layers_n_, std::vector<float>(static_cast<size_t>(max_len_) * d_));
  c.v = c.k;
  return c;
}

void Model::forward(Cache& c, const std::vector<int>& toks, float* logits, bool avx2) const {
  const int T = static_cast<int>(toks.size()), d = d_, p0 = c.len, dh = d / heads_;
  if (p0 + T > max_len_) throw std::runtime_error("sequence exceeds max_len");
  std::vector<float> x(static_cast<size_t>(T) * d), h(x.size()), qkv(static_cast<size_t>(T) * 3 * d),
      att(x.size()), proj(x.size()), f(static_cast<size_t>(T) * ff_);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < d; ++i) {
      x[t * d + i] = tok_emb_[toks[t] * d + i] + pos_emb_[(p0 + t) * d + i];
    }
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
  for (int l = 0; l < layers_n_; ++l) {
    const Layer& L = layers_[l];
    layernorm(x.data(), L.ln1_g, L.ln1_b, T, d, h.data());
    linear(L.qkv, h.data(), T, qkv.data(), avx2);
    for (int t = 0; t < T; ++t) {
      std::memcpy(&c.k[l][(p0 + t) * d], &qkv[t * 3 * d + d], d * sizeof(float));
      std::memcpy(&c.v[l][(p0 + t) * d], &qkv[t * 3 * d + 2 * d], d * sizeof(float));
    }
    std::vector<float> score(p0 + T);
    for (int t = 0; t < T; ++t) {
      const int q_pos = p0 + t;
      for (int hd = 0; hd < heads_; ++hd) {
        const float* q = &qkv[t * 3 * d + hd * dh];
        float mx = -INFINITY;
        for (int s = 0; s <= q_pos; ++s) {
          score[s] = dot_f32(q, &c.k[l][s * d + hd * dh], dh) * scale;
          if (score[s] > mx) mx = score[s];
        }
        float sum = 0.0f;
        for (int s = 0; s <= q_pos; ++s) {
          score[s] = std::exp(score[s] - mx);
          sum += score[s];
        }
        float* out = &att[t * d + hd * dh];
        for (int i = 0; i < dh; ++i) out[i] = 0.0f;
        for (int s = 0; s <= q_pos; ++s) {
          const float w = score[s] / sum;
          const float* v = &c.v[l][s * d + hd * dh];
          for (int i = 0; i < dh; ++i) out[i] += w * v[i];
        }
      }
    }
    linear(L.proj, att.data(), T, proj.data(), avx2);
    for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];
    layernorm(x.data(), L.ln2_g, L.ln2_b, T, d, h.data());
    linear(L.fc1, h.data(), T, f.data(), avx2);
    for (float& v : f) v = gelu(v);
    linear(L.fc2, f.data(), T, proj.data(), avx2);
    for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];
  }
  c.len += T;
  layernorm(x.data() + static_cast<size_t>(T - 1) * d, lnf_g_, lnf_b_, 1, d, h.data());
  linear(head_, h.data(), 1, logits, avx2);
}
