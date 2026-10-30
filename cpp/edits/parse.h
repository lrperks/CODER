#pragma once
#include <string>
#include <vector>

#include "tree_sitter/api.h"

// A parsed Python source. Node ids are indices into nodes(), which lists every
// node (named and anonymous) in pre-order, so ids are stable within one parse.
class Parsed {
 public:
  explicit Parsed(std::string src);
  ~Parsed();
  Parsed(const Parsed&) = delete;
  Parsed& operator=(const Parsed&) = delete;

  const std::string& src() const { return src_; }
  const std::vector<TSNode>& nodes() const { return nodes_; }
  bool valid() const;  // false if any ERROR or MISSING node exists
  int start(TSNode n) const { return static_cast<int>(ts_node_start_byte(n)); }
  int end(TSNode n) const { return static_cast<int>(ts_node_end_byte(n)); }
  std::string text(TSNode n) const { return src_.substr(start(n), end(n) - start(n)); }

 private:
  void collect(TSNode n);

  std::string src_;
  TSTree* tree_ = nullptr;
  std::vector<TSNode> nodes_;
};
