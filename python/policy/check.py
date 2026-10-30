"""Checks the C++ runtime against the PyTorch reference on fixed prompts."""
import json
import random
import re
from pathlib import Path

import vedit
from python.data.dsl import PROGRAM
from python.policy import reference
from python.policy.export import dequantized  # noqa: F401  (kept for callers that need it)

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data" / "samples.jsonl"


def heldout_prompts(name, n=100, seed=0):
    """n fixed held-out prompts: annotated for the edit policy, plain for the baseline."""
    rows = [json.loads(line) for line in SAMPLES.read_text().splitlines()]
    rows = [r for r in rows if r["split"] == "heldout"]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    return [vedit.render_prompt(r["test"], r["buggy"], name == "edit").encode() for r in rows]


def max_new_tokens(name):
    """Generation budget. The baseline may use the context left after the longest held-out prompt."""
    if name == "edit":
        return 160
    rows = [json.loads(line) for line in SAMPLES.read_text().splitlines()]
    longest = max(len(vedit.render_prompt(r["test"], r["buggy"], False))
                  for r in rows if r["split"] == "heldout")
    return reference_max_len() - longest


def reference_max_len():
    from python.policy.model import CFG
    return CFG["max_len"]


def parity(name, n=100):
    """Greedy outputs must match token-for-token. Returns (mismatches, total)."""
    model_path = ROOT / "results" / "models" / f"{name}.bin"
    cpp = vedit.Model(str(model_path))
    ref = reference.load_reference(name)
    edit = name == "edit"
    mode = vedit.Mode.Edit if edit else vedit.Mode.Free
    max_new = max_new_tokens(name)
    bad = []
    prompts = heldout_prompts(name, n)
    for i, p in enumerate(prompts):
        got, _, _ = vedit.decode(cpp, p.decode(), mode, max_new, 0.0, 0, True, True)
        want, _ = reference.decode(ref, p, edit, max_new)
        if got != want:
            bad.append({"prompt": i, "cpp": got, "ref": want})
    return bad, len(prompts)


def grammar_validity(name, n=100):
    """Fraction of greedy edit programs that match the DSL (C++ NFA and Python re)."""
    cpp = vedit.Model(str(ROOT / "results" / "models" / f"{name}.bin"))
    ok_cpp = ok_re = 0
    prompts = heldout_prompts(name, n)
    for p in prompts:
        text, _, _ = vedit.decode(cpp, p.decode(), vedit.Mode.Edit, 160, 0.0, 0, True, True)
        ok_cpp += vedit.valid_program(text)
        ok_re += re.fullmatch(PROGRAM, text) is not None
    return ok_cpp / len(prompts), ok_re / len(prompts)
