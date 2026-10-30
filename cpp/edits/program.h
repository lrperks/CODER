#pragma once
#include <optional>
#include <string>
#include <vector>

enum class Kind { Noop, Replace, SwapOp, ChangeReturn, AddGuard, InsertBefore, DeleteStmt, SwapArgs };

struct Prim {
  Kind kind = Kind::Noop;
  int id = 0;
  std::string text;  // new source, operator, or guard/statement text
  int i = 0, j = 0;  // swap_args indices
};

// Parses DSL text such as "swap_op(012,<=);delete_stmt(004)\n". Returns nullopt
// if the text is not a valid program under the grammar in runtime/grammar.h.
std::optional<std::vector<Prim>> parse_program(const std::string& text);
