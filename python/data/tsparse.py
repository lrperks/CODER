"""Parse Python source with tree-sitter (same grammar as the C++ runtime)."""
import tree_sitter as ts
import tree_sitter_python as tsp

_PARSER = ts.Parser(ts.Language(tsp.language()))


def parse(src: str):
    """Return (root, nodes); nodes lists every node in pre-order, so index == node id."""
    root = _PARSER.parse(src.encode()).root_node
    nodes, stack = [], [root]
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(reversed(n.children))
    return root, nodes
