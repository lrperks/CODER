"""Mutation types over one fixture function.

Each proposal is a set of text edits in fixed-source byte offsets plus an undo
descriptor. generate.py resolves the descriptor to DSL node ids in the buggy parse.
"""
from dataclasses import dataclass
import re

ARITH = ["+", "-", "*", "/", "//", "%"]
COMPARE = ["<", "<=", ">", ">=", "==", "!="]
GUARD_RE = re.compile(r"if (.+) is None: return None")


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Proposal:
    name: str
    edits: tuple
    undo: dict  # kind, pos, after, span, old/cond/text as needed


def _line_bounds(src, start, end):
    ls = src.rfind("\n", 0, start) + 1
    le = src.find("\n", end)
    return ls, (len(src) if le < 0 else le + 1)


def _single_line_stmts(src, nodes):
    """Statements that sit alone on their line inside a block and have a following sibling."""
    for n in nodes:
        if n.parent is None or n.parent.type != "block" or not n.is_named:
            continue
        if "\n" in n.text.decode() or n.next_named_sibling is None:
            continue
        ls, le = _line_bounds(src, n.start_byte, n.end_byte)
        if src[ls:le].strip() == n.text.decode():
            yield n, ls, le


def proposals(src, nodes):
    out = []
    names = {n.text.decode() for n in nodes if n.type == "identifier"}
    for n in nodes:
        t = n.type
        if t == "integer":
            v = n.text.decode()
            new = str(int(v) + 1)
            out.append(Proposal("off_by_one", (Edit(n.start_byte, n.end_byte, new),),
                                {"kind": "replace", "pos": n.start_byte, "after": False,
                                 "span": len(new), "old": v}))
        elif t == "binary_operator":
            for c in n.children:
                if not c.is_named and c.type in ARITH:
                    for new in ARITH:
                        if new != c.type:
                            out.append(Proposal("wrong_arith_op", (Edit(c.start_byte, c.end_byte, new),),
                                                {"kind": "swap_op", "pos": c.start_byte, "after": False,
                                                 "span": len(new), "old": c.type}))
        elif t == "comparison_operator":
            for c in n.children:
                if not c.is_named and c.type in COMPARE:
                    for new in COMPARE:
                        if new != c.type:
                            out.append(Proposal("wrong_compare_op", (Edit(c.start_byte, c.end_byte, new),),
                                                {"kind": "swap_op", "pos": c.start_byte, "after": False,
                                                 "span": len(new), "old": c.type}))
        elif t == "call":
            args = n.child_by_field_name("arguments")
            if args is None:
                continue
            pos_args = [a for a in args.named_children if a.type != "keyword_argument"]
            if len(pos_args) == 2 and pos_args[0].text != pos_args[1].text and len(args.named_children) == 2:
                a, b = pos_args
                out.append(Proposal("swapped_args",
                                    (Edit(a.start_byte, a.end_byte, b.text.decode()),
                                     Edit(b.start_byte, b.end_byte, a.text.decode())),
                                    {"kind": "swap_args", "pos": n.start_byte, "after": True}))
        elif t == "return_statement" and n.named_child_count == 1:
            expr = n.named_children[0]
            if expr.text.decode() != "None":
                out.append(Proposal("wrong_return", (Edit(expr.start_byte, n.end_byte, "None"),),
                                    {"kind": "change_return", "pos": n.start_byte, "after": True,
                                     "old": src[expr.start_byte:n.end_byte]}))
        elif t == "identifier" and n.parent is not None and n.parent.type not in ("attribute", "function_definition"):
            for other in sorted(names - {n.text.decode()}):
                out.append(Proposal("wrong_var", (Edit(n.start_byte, n.end_byte, other),),
                                    {"kind": "replace", "pos": n.start_byte, "after": False,
                                     "span": len(other), "old": n.text.decode()}))
        elif t == "if_statement" and n.child_by_field_name("condition") is not None:
            cond = n.child_by_field_name("condition")
            new = f"not ({cond.text.decode()})"
            out.append(Proposal("negate_condition", (Edit(cond.start_byte, cond.end_byte, new),),
                                {"kind": "replace", "pos": cond.start_byte, "after": False,
                                 "span": len(new), "old": cond.text.decode()}))
    out.extend(_stmt_proposals(src, nodes))
    return out


def _stmt_proposals(src, nodes):
    out = []
    for n, ls, le in _single_line_stmts(src, nodes):
        nxt = n.next_named_sibling
        if n.type == "if_statement":
            g = GUARD_RE.fullmatch(n.text.decode())
            if g:
                out.append(Proposal("missing_none_guard", (Edit(ls, le, ""),),
                                    {"kind": "add_guard", "pos": nxt.start_byte, "after": True,
                                     "old": g.group(1) + " is None"}))
        if n.type not in ("if_statement", "for_statement", "while_statement", "function_definition"):
            out.append(Proposal("deleted_stmt", (Edit(ls, le, ""),),
                                {"kind": "insert_before", "pos": nxt.start_byte, "after": True,
                                 "old": n.text.decode()}))
    for n in nodes:
        if n.parent is not None and n.parent.type == "block" and n.is_named and n.type != "function_definition":
            if "\n" in n.text.decode():
                continue
            ls = src.rfind("\n", 0, n.start_byte) + 1
            ind = src[ls:n.start_byte]
            if ind.strip() == "":
                out.append(Proposal("early_return", (Edit(ls, ls, ind + "return None\n"),),
                                    {"kind": "delete_stmt", "pos": ls, "after": False, "offset": len(ind)}))
    return out
