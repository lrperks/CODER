#include "edits/edits.h"

#include <algorithm>

#include "edits/parse.h"

namespace {

struct Edit {
  int start, end;
  std::string text;
};
using Edits = std::optional<std::vector<Edit>>;

int line_start_of(const std::string& src, int pos) {
  while (pos > 0 && src[pos - 1] != '\n') --pos;
  return pos;
}

bool blank(const std::string& s, int from, int to) {
  for (int k = from; k < to; ++k) {
    if (s[k] != ' ' && s[k] != '\t') return false;
  }
  return true;
}

Edits edits_for(const Parsed& P, const Prim& p) {
  const std::string& src = P.src();
  if (p.kind == Kind::Noop) return std::vector<Edit>{};
  if (p.id < 0 || p.id >= static_cast<int>(P.nodes().size())) return std::nullopt;
  TSNode n = P.nodes()[p.id];
  int s = P.start(n), e = P.end(n);
  std::string type = ts_node_type(n);
  switch (p.kind) {
    case Kind::Replace:
      return std::vector<Edit>{{s, e, p.text}};
    case Kind::SwapOp:
      if (ts_node_is_named(n)) return std::nullopt;
      return std::vector<Edit>{{s, e, p.text}};
    case Kind::ChangeReturn: {
      if (type != "return_statement") return std::nullopt;
      int kw_end = static_cast<int>(ts_node_end_byte(ts_node_child(n, 0)));
      return std::vector<Edit>{{kw_end, e, " " + p.text}};
    }
    case Kind::AddGuard:
    case Kind::InsertBefore: {
      if (!ts_node_is_named(n)) return std::nullopt;
      int ls = line_start_of(src, s);
      if (!blank(src, ls, s)) return std::nullopt;
      std::string indent = src.substr(ls, s - ls);
      std::string body = p.kind == Kind::AddGuard ? "if " + p.text + ": return None" : p.text;
      return std::vector<Edit>{{ls, ls, indent + body + "\n"}};
    }
    case Kind::DeleteStmt: {
      if (!ts_node_is_named(n) || src.substr(s, e - s).find('\n') != std::string::npos) return std::nullopt;
      int ls = line_start_of(src, s);
      if (!blank(src, ls, s)) return std::nullopt;
      int le = e, size = static_cast<int>(src.size());
      while (le < size && (src[le] == ' ' || src[le] == '\t')) ++le;
      if (le < size && src[le] != '\n') return std::nullopt;
      return std::vector<Edit>{{ls, le < size ? le + 1 : le, ""}};
    }
    case Kind::SwapArgs: {
      if (type != "call") return std::nullopt;
      TSNode args = ts_node_child_by_field_name(n, "arguments", 9);
      if (ts_node_is_null(args)) return std::nullopt;
      uint32_t count = ts_node_named_child_count(args);
      if (p.i == p.j || p.i >= static_cast<int>(count) || p.j >= static_cast<int>(count))
        return std::nullopt;
      TSNode a = ts_node_named_child(args, p.i), b = ts_node_named_child(args, p.j);
      return std::vector<Edit>{{P.start(a), P.end(a), P.text(b)}, {P.start(b), P.end(b), P.text(a)}};
    }
    case Kind::Noop:
      break;
  }
  return std::nullopt;
}

}  // namespace

std::optional<std::string> apply_program(const std::string& src, const std::vector<Prim>& prog) {
  Parsed P(src);
  std::vector<Edit> all;
  for (const Prim& p : prog) {
    Edits es = edits_for(P, p);
    if (!es) return std::nullopt;
    all.insert(all.end(), es->begin(), es->end());
  }
  std::stable_sort(all.begin(), all.end(), [](const Edit& a, const Edit& b) {
    return a.start < b.start || (a.start == b.start && a.end < b.end);
  });
  std::string out;
  int pos = 0;
  for (const Edit& e : all) {
    if (e.start < pos) return std::nullopt;  // overlapping edits
    out.append(src, pos, e.start - pos);
    out += e.text;
    pos = e.end;
  }
  out.append(src, pos, std::string::npos);
  if (!Parsed(out).valid()) return std::nullopt;
  return out;
}
