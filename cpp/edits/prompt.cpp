#include "edits/prompt.h"

#include <cstdio>
#include <set>

#include "edits/parse.h"

namespace {

bool annotated_type(TSNode n) {
  static const std::set<std::string> ops = {"+", "-", "*", "/", "//", "%", "<", "<=", ">",
                                            ">=", "==", "!="};
  return ts_node_is_named(n) || ops.count(ts_node_type(n)) > 0;
}

std::string annotate(const std::string& src) {
  Parsed P(src);
  std::vector<std::string> marks(src.size() + 1);
  const auto& nodes = P.nodes();
  for (size_t id = 0; id < nodes.size(); ++id) {  // pre-order: outer marks come first
    int s = P.start(nodes[id]), e = P.end(nodes[id]);
    if (s >= e || !annotated_type(nodes[id])) continue;
    char buf[16];
    std::snprintf(buf, sizeof buf, "{%03zu}", id);
    marks[s] += buf;
  }
  std::string out;
  for (size_t i = 0; i < src.size(); ++i) {
    out += marks[i];
    out += src[i];
  }
  return out + marks[src.size()];
}

std::string strip_newlines(std::string s) {
  while (!s.empty() && s.back() == '\n') s.pop_back();
  return s;
}

}  // namespace

std::string render_prompt(const std::string& test, const std::string& src, bool annotated) {
  std::string body = annotated ? annotate(src) : src;
  if (body.empty() || body.back() != '\n') body += '\n';
  return strip_newlines(test) + "\n#\n" + body + ">";
}
