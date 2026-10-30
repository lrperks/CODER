#include "runtime/nfa.h"

#include <cctype>
#include <stdexcept>

namespace {

struct Node {
  enum Type { Chars, Cat, Alt, Rep } type;
  std::bitset<128> chars;
  std::vector<Node> kids;
  int min = 1, max = 1;  // max < 0: unbounded
};

class Parser {
 public:
  explicit Parser(std::string p) : p_(std::move(p)) {}

  Node parse() {
    Node n = alt();
    if (i_ != p_.size()) throw std::runtime_error("regex: unbalanced ')'");
    return n;
  }

 private:
  bool at(char c) const { return i_ < p_.size() && p_[i_] == c; }

  unsigned char next_char() {
    unsigned char c = p_.at(i_++);
    if (c == '\\') {
      c = p_.at(i_++);
      if (c == 'n') c = '\n';
    }
    return c;
  }

  Node alt() {
    Node n{Node::Alt, {}, {}, 1, 1};
    n.kids.push_back(cat());
    while (at('|')) {
      ++i_;
      n.kids.push_back(cat());
    }
    return n.kids.size() == 1 ? n.kids[0] : n;
  }

  Node cat() {
    Node n{Node::Cat, {}, {}, 1, 1};
    while (i_ < p_.size() && !at('|') && !at(')')) n.kids.push_back(rep());
    return n;
  }

  int number() {
    int v = 0;
    while (i_ < p_.size() && std::isdigit(static_cast<unsigned char>(p_[i_])))
      v = v * 10 + (p_[i_++] - '0');
    return v;
  }

  Node rep() {
    Node a = atom();
    while (i_ < p_.size() && std::string("*+?{").find(p_[i_]) != std::string::npos) {
      char q = p_[i_++];
      int mn = 0, mx = -1;
      if (q == '+') mn = 1;
      if (q == '?') mx = 1;
      if (q == '{') {
        mn = mx = number();
        if (at(',')) {
          ++i_;
          mx = at('}') ? -1 : number();
        }
        if (!at('}')) throw std::runtime_error("regex: bad repeat");
        ++i_;
      }
      a = Node{Node::Rep, {}, {a}, mn, mx};
    }
    return a;
  }

  Node atom() {
    if (at('(')) {
      ++i_;
      Node n = alt();
      if (!at(')')) throw std::runtime_error("regex: missing ')'");
      ++i_;
      return n;
    }
    Node n{Node::Chars, {}, {}, 1, 1};
    if (at('[')) {
      ++i_;
      n.chars = charclass();
      return n;
    }
    n.chars.set(next_char());
    return n;
  }

  std::bitset<128> charclass() {
    std::bitset<128> set;
    while (!at(']')) {
      unsigned char lo = next_char();
      if (at('-') && i_ + 1 < p_.size() && p_[i_ + 1] != ']') {
        ++i_;
        unsigned char hi = next_char();
        for (unsigned c = lo; c <= hi; ++c) set.set(c);
      } else {
        set.set(lo);
      }
    }
    ++i_;
    return set;
  }

  std::string p_;
  size_t i_ = 0;
};

int add(std::vector<Nfa::State>& st, Nfa::State s) {
  st.push_back(std::move(s));
  return static_cast<int>(st.size()) - 1;
}

int build(std::vector<Nfa::State>& st, const Node& n, int next) {
  switch (n.type) {
    case Node::Chars: {
      Nfa::State s;
      s.chars = n.chars;
      s.is_char = true;
      s.out = next;
      return add(st, s);
    }
    case Node::Cat: {
      int cur = next;
      for (auto it = n.kids.rbegin(); it != n.kids.rend(); ++it) cur = build(st, *it, cur);
      return cur;
    }
    case Node::Alt: {
      Nfa::State s;
      for (const Node& k : n.kids) s.eps.push_back(build(st, k, next));
      return add(st, s);
    }
    case Node::Rep: {
      int tail = next;
      if (n.max < 0) {
        int loop = add(st, Nfa::State{});
        int body = build(st, n.kids[0], loop);
        st[loop].eps = {body, next};
        tail = loop;
      }
      for (int k = 0; k < n.max - n.min; ++k) {
        int opt = add(st, Nfa::State{});
        int body = build(st, n.kids[0], tail);
        st[opt].eps = {body, tail};
        tail = opt;
      }
      for (int k = 0; k < n.min; ++k) tail = build(st, n.kids[0], tail);
      return tail;
    }
  }
  return next;
}

}  // namespace

Nfa::Nfa(const std::string& pattern) {
  Node root = Parser(pattern).parse();
  accept_ = add(states_, State{});
  start_ = build(states_, root, accept_);
}

Nfa::States Nfa::closure(States seeds) const {
  std::vector<char> seen(states_.size(), 0);
  States out;
  while (!seeds.empty()) {
    int s = seeds.back();
    seeds.pop_back();
    if (seen[s]) continue;
    seen[s] = 1;
    out.push_back(s);
    for (int e : states_[s].eps) seeds.push_back(e);
  }
  return out;
}

Nfa::States Nfa::start() const { return closure({start_}); }

Nfa::States Nfa::step(const States& s, char c) const {
  if (static_cast<unsigned char>(c) & 0x80) return {};
  States next;
  for (int i : s) {
    const State& st = states_[i];
    if (st.is_char && st.chars.test(static_cast<unsigned char>(c))) next.push_back(st.out);
  }
  return closure(next);
}

std::bitset<128> Nfa::allowed(const States& s) const {
  std::bitset<128> bits;
  for (int i : s) {
    if (states_[i].is_char) bits |= states_[i].chars;
  }
  if (accepting(s)) bits.set(0);
  return bits;
}

bool Nfa::accepting(const States& s) const {
  for (int i : s) {
    if (i == accept_) return true;
  }
  return false;
}

bool Nfa::matches(const std::string& text) const {
  States s = start();
  for (char c : text) {
    s = step(s, c);
    if (s.empty()) return false;
  }
  return accepting(s);
}
