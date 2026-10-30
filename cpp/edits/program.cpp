#include "edits/program.h"

#include "runtime/grammar.h"

namespace {

Prim parse_step(const std::string& s) {
  size_t open = s.find('(');
  std::string name = s.substr(0, open);
  std::string args = s.substr(open + 1, s.size() - open - 2);
  Prim p;
  if (name == "noop") return p;
  p.id = std::stoi(args.substr(0, 3));
  if (name == "delete_stmt") {
    p.kind = Kind::DeleteStmt;
  } else if (name == "swap_args") {
    p.kind = Kind::SwapArgs;
    p.i = args[4] - '0';
    p.j = args[6] - '0';
  } else {
    p.text = args.substr(4);
    if (name == "swap_op") p.kind = Kind::SwapOp;
    if (name == "replace") p.kind = Kind::Replace;
    if (name == "change_return") p.kind = Kind::ChangeReturn;
    if (name == "add_guard") p.kind = Kind::AddGuard;
    if (name == "insert_before") p.kind = Kind::InsertBefore;
  }
  return p;
}

}  // namespace

std::optional<std::vector<Prim>> parse_program(const std::string& text) {
  if (!program_nfa().matches(text)) return std::nullopt;
  std::vector<Prim> prog;
  std::string body = text.substr(0, text.size() - 1);  // drop the final newline
  size_t pos = 0;
  while (pos <= body.size()) {
    size_t semi = body.find(';', pos);
    if (semi == std::string::npos) semi = body.size();
    prog.push_back(parse_step(body.substr(pos, semi - pos)));
    pos = semi + 1;
  }
  return prog;
}
