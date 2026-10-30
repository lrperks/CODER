#pragma once
#include <bitset>
#include <string>
#include <vector>

// Thompson NFA over 7-bit characters for a small regex subset: literals,
// \escapes, [classes] with ranges, (), |, *, +, ?, {m}, {m,n}.
// Bit 0 of allowed() stands for end-of-program (EOS) when accepting.
class Nfa {
 public:
  using States = std::vector<int>;
  struct State {
    std::bitset<128> chars;
    bool is_char = false;
    int out = -1;
    std::vector<int> eps;
  };

  explicit Nfa(const std::string& pattern);
  States start() const;
  States step(const States& s, char c) const;
  std::bitset<128> allowed(const States& s) const;
  bool accepting(const States& s) const;
  bool matches(const std::string& text) const;

 private:
  States closure(States seeds) const;

  std::vector<State> states_;
  int start_ = -1;
  int accept_ = -1;
};
