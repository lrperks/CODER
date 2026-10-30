#include <gtest/gtest.h>

#include "edits/edits.h"
#include "edits/parse.h"

namespace {

const char* kSrc = "def f(a, b):\n    if a > b:\n        return a\n    return pow(a, 2) + b\n";

// First node of the given type whose text equals `text`.
int find(const Parsed& P, const char* type, const std::string& text) {
  for (size_t i = 0; i < P.nodes().size(); ++i) {
    TSNode n = P.nodes()[i];
    if (ts_node_type(n) == std::string(type) && P.text(n) == text) return static_cast<int>(i);
  }
  return -1;
}

Prim prim(Kind k, int id, std::string text = "", int i = 0, int j = 0) {
  Prim p;
  p.kind = k;
  p.id = id;
  p.text = std::move(text);
  p.i = i;
  p.j = j;
  return p;
}

}  // namespace

TEST(Edits, ParsedSourceIsValid) {
  Parsed P(kSrc);
  EXPECT_TRUE(P.valid());
  EXPECT_EQ(ts_node_type(P.nodes()[0]), std::string("module"));
}

TEST(Edits, ReplaceInteger) {
  Parsed P(kSrc);
  auto out = apply_program(kSrc, {prim(Kind::Replace, find(P, "integer", "2"), "3")});
  ASSERT_TRUE(out);
  EXPECT_NE(out->find("pow(a, 3)"), std::string::npos);
}

TEST(Edits, SwapOperatorOnComparison) {
  Parsed P(kSrc);
  int op = find(P, ">", ">");
  auto out = apply_program(kSrc, {prim(Kind::SwapOp, op, ">=")});
  ASSERT_TRUE(out);
  EXPECT_NE(out->find("a >= b"), std::string::npos);
}

TEST(Edits, ChangeReturnReplacesExpression) {
  Parsed P(kSrc);
  int ret = find(P, "return_statement", "return a");
  auto out = apply_program(kSrc, {prim(Kind::ChangeReturn, ret, "b")});
  ASSERT_TRUE(out);
  EXPECT_NE(out->find("        return b\n"), std::string::npos);
}

TEST(Edits, DeleteStatementRemovesLine) {
  Parsed P(kSrc);
  int ret = find(P, "return_statement", "return a");
  auto out = apply_program(kSrc, {prim(Kind::DeleteStmt, ret)});
  ASSERT_TRUE(out);
  EXPECT_EQ(out->find("return a"), std::string::npos);
}

TEST(Edits, InsertBeforeKeepsIndent) {
  Parsed P(kSrc);
  int ret = find(P, "return_statement", "return pow(a, 2) + b");
  auto out = apply_program(kSrc, {prim(Kind::InsertBefore, ret, "x = 1")});
  ASSERT_TRUE(out);
  EXPECT_NE(out->find("    x = 1\n    return pow"), std::string::npos);
}

TEST(Edits, AddGuardInsertsNoneGuard) {
  Parsed P(kSrc);
  int ret = find(P, "return_statement", "return pow(a, 2) + b");
  auto out = apply_program(kSrc, {prim(Kind::AddGuard, ret, "a is None")});
  ASSERT_TRUE(out);
  EXPECT_NE(out->find("    if a is None: return None\n    return pow"), std::string::npos);
}

TEST(Edits, SwapArgsOnCall) {
  Parsed P(kSrc);
  int call = find(P, "call", "pow(a, 2)");
  auto out = apply_program(kSrc, {prim(Kind::SwapArgs, call, "", 0, 1)});
  ASSERT_TRUE(out);
  EXPECT_NE(out->find("pow(2, a)"), std::string::npos);
}

TEST(Edits, RejectsEditThatBreaksParse) {
  Parsed P(kSrc);
  EXPECT_FALSE(apply_program(kSrc, {prim(Kind::Replace, find(P, "integer", "2"), "(")}));
}

TEST(Edits, RejectsBadIdsAndWrongTypes) {
  Parsed P(kSrc);
  EXPECT_FALSE(apply_program(kSrc, {prim(Kind::Replace, 99999, "1")}));
  EXPECT_FALSE(apply_program(kSrc, {prim(Kind::SwapOp, find(P, "identifier", "a"), "+")}));
  EXPECT_FALSE(apply_program(kSrc, {prim(Kind::ChangeReturn, find(P, "call", "pow(a, 2)"), "b")}));
}

TEST(Edits, IndependentPrimitivesCompose) {
  Parsed P(kSrc);
  auto out = apply_program(kSrc, {prim(Kind::Replace, find(P, "integer", "2"), "3"),
                                  prim(Kind::SwapOp, find(P, ">", ">"), "<")});
  ASSERT_TRUE(out);
  EXPECT_NE(out->find("a < b"), std::string::npos);
  EXPECT_NE(out->find("pow(a, 3)"), std::string::npos);
}

TEST(Edits, NoopReturnsSourceUnchanged) {
  auto out = apply_program(kSrc, {prim(Kind::Noop, 0)});
  ASSERT_TRUE(out);
  EXPECT_EQ(*out, kSrc);
}
