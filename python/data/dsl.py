"""DSL shared by the generator and the C++ grammar (same pattern string)."""
import re

ID = "[0-9]{3}"
SNIP_CHARS = "A-Za-z0-9 _.+\\-*/%<>=!:,\\[\\]'\"&|^~"  # no parens, ';' or newline
SNIP = f"[{SNIP_CHARS}]+"
OP = "[-+*/%<>=!]{1,2}"
STEP = (
    r"(noop\(\)"
    rf"|swap_op\({ID},{OP}\)"
    rf"|replace\({ID},{SNIP}\)"
    rf"|change_return\({ID},{SNIP}\)"
    rf"|add_guard\({ID},{SNIP}\)"
    rf"|insert_before\({ID},{SNIP}\)"
    rf"|delete_stmt\({ID}\)"
    r"|swap_args\(" + ID + r",[0-9],[0-9]\))"
)
PROGRAM = STEP + r"(;" + STEP + r"){0,2}\n"
SNIP_RE = re.compile(SNIP)


def snippet_ok(text: str) -> bool:
    return SNIP_RE.fullmatch(text) is not None


def step(kind: str, *args) -> str:
    """Format one DSL step. kind is the primitive name, args are (id, payload?)."""
    if kind == "delete_stmt":
        return f"delete_stmt({args[0]:03d})"
    if kind == "swap_args":
        return f"swap_args({args[0]:03d},{args[1]},{args[2]})"
    return f"{kind}({args[0]:03d},{args[1]})"
