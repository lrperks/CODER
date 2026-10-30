import json
from pathlib import Path

import pytest

from python.data.dsl import PROGRAM

vedit = pytest.importorskip("vedit")
OUT = Path(__file__).resolve().parents[2] / "data" / "samples.jsonl"


def test_pattern_matches_python_dsl():
    assert vedit.PROGRAM_PATTERN == PROGRAM


def test_grammar_accepts_and_rejects():
    assert vedit.valid_program("replace(012,x + 1)\n")
    assert not vedit.valid_program("replace(12,x)\n")
    assert not vedit.valid_program("noop()")


def test_gold_programs_reproduce_fixed_source():
    if not OUT.exists():
        pytest.skip("run python -m python.data.generate first")
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    bad = [r["id"] for r in rows if vedit.apply_program(r["buggy"], r["gold"]) != r["fixed"]]
    assert not bad, f"{len(bad)} gold programs do not reproduce the fixed source: {bad[:10]}"


def test_prompt_marks_nodes():
    p = vedit.render_prompt("def test_f():\n    assert f(1) == 2\n", "def f(x):\n    return x + 1\n", True)
    assert "{001}" in p and p.endswith(">")
