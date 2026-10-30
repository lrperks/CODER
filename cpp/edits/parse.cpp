#include "edits/parse.h"

#include <utility>

extern "C" const TSLanguage* tree_sitter_python();

Parsed::Parsed(std::string src) : src_(std::move(src)) {
  TSParser* parser = ts_parser_new();
  ts_parser_set_language(parser, tree_sitter_python());
  tree_ = ts_parser_parse_string(parser, nullptr, src_.data(), static_cast<uint32_t>(src_.size()));
  ts_parser_delete(parser);
  collect(ts_tree_root_node(tree_));
}

Parsed::~Parsed() { ts_tree_delete(tree_); }

void Parsed::collect(TSNode n) {
  nodes_.push_back(n);
  uint32_t count = ts_node_child_count(n);
  for (uint32_t i = 0; i < count; ++i) collect(ts_node_child(n, i));
}

bool Parsed::valid() const { return !ts_node_has_error(ts_tree_root_node(tree_)); }
