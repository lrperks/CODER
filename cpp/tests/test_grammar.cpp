#include <gtest/gtest.h>

#include "edits/program.h"
#include "runtime/grammar.h"

namespace {

std::string allowed_chars(const std::string& prefix) {
  const Nfa& nfa = program_nfa();
  Nfa::States s = nfa.start();
  for (char c : prefix) s = nfa.step(s, c);
  auto bits = nfa.allowed(s);
  std::string out;
  for (int c = 0; c < 128; ++c) {
    if (bits.test(c)) out += static_cast<char>(c == 0 ? '$' : c);
  }
  return out;
}

}  // namespace

TEST(Grammar, AcceptsValidPrograms) {
  const Nfa& nfa = program_nfa();
  EXPECT_TRUE(nfa.matches("noop()\n"));
  EXPECT_TRUE(nfa.matches("replace(012,x + 1)\n"));
  EXPECT_TRUE(nfa.matches("swap_op(001,<=)\n"));
  EXPECT_TRUE(nfa.matches("swap_args(003,0,1);delete_stmt(004);add_guard(005,xs is None)\n"));
}

TEST(Grammar, RejectsInvalidPrograms) {
  const Nfa& nfa = program_nfa();
  EXPECT_FALSE(nfa.matches("noop()"));          // no terminating newline
  EXPECT_FALSE(nfa.matches("replace(12,x)\n")); // ids are 3 digits
  EXPECT_FALSE(nfa.matches("replace(012,f(x))\n"));  // parens are not in snippets
  EXPECT_FALSE(nfa.matches("noop();noop();noop();noop()\n"));  // at most 3 steps
}

TEST(Grammar, ParseProgramRoundTrip) {
  auto prog = parse_program("swap_args(003,0,1);replace(012,x + 1)\n");
  ASSERT_TRUE(prog);
  ASSERT_EQ(prog->size(), 2u);
  EXPECT_EQ((*prog)[0].kind, Kind::SwapArgs);
  EXPECT_EQ((*prog)[0].id, 3);
  EXPECT_EQ((*prog)[1].kind, Kind::Replace);
  EXPECT_EQ((*prog)[1].text, "x + 1");
}

TEST(Grammar, AllowedSetsAndForcedChars) {
  EXPECT_EQ(allowed_chars("delete_stmt("), "0123456789");
  EXPECT_EQ(allowed_chars("noop()"), "\n;");
  EXPECT_EQ(allowed_chars("replace"), "(");  // forced continuation
  EXPECT_EQ(allowed_chars("noop()\n"), "$");   // complete program: EOS only
}
