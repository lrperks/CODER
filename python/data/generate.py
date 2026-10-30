"""Generate verified-edit training samples: (buggy, failing test, fixed, gold program).

Usage: python -m python.data.generate --count 4000 --seed 0
"""
import argparse
import json
import random
import signal
from pathlib import Path

from python.data import dsl
from python.data.mutate import Edit, proposals
from python.data.tsparse import parse

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
OUT = ROOT / "data" / "samples.jsonl"
SEP = "# --- test ---\n"


def load_fixtures():
    out = []
    for p in sorted(FIXTURES.glob("*.py")):
        fixed, test = p.read_text().split(SEP)
        out.append((p.stem, fixed, test))
    return out


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout()


def passes(src: str, test: str) -> bool:
    """True if the test passes on src (in-process, 1 s cap; timeouts count as failures)."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(1)
    try:
        ns = {}
        exec(compile(src, "<candidate>", "exec"), ns)
        exec(compile(test, "<test>", "exec"), ns)
        ns["test_f"]()
        return True
    except BaseException:
        return False
    finally:
        signal.alarm(0)


def apply_edits(src: str, edits) -> str:
    out, pos = [], 0
    for e in sorted(edits, key=lambda e: (e.start, e.end)):
        out.append(src[pos:e.start])
        out.append(e.text)
        pos = e.end
    out.append(src[pos:])
    return "".join(out)


def conflicts(a, b) -> bool:
    for x in a:
        for y in b:
            if x.start < y.end and y.start < x.end:
                return True
            if x.start == y.start and (x.start == x.end or y.start == y.end):
                return True
    return False


def to_buggy(pos, edits, after):
    """Map a fixed-source offset to the buggy source. `after` decides insertions at pos."""
    d = 0
    for e in edits:
        if e.end <= pos and not (e.start == e.end == pos and not after):
            d += len(e.text) - (e.end - e.start)
    return pos + d


def gold_steps(buggy, edits, props):
    _, nodes = parse(buggy)
    steps = []
    for p in props:
        u = p.undo
        pos = to_buggy(u["pos"], edits, u["after"]) + u.get("offset", 0)
        kind = u["kind"]
        if kind in ("replace", "swap_op"):
            end = pos + u["span"]
            hits = [n for n in nodes if n.start_byte == pos and n.end_byte == end
                    and (kind == "replace" or not n.is_named)]
            if not hits:
                return None
            nid = nodes.index(hits[-1])
            steps.append(dsl.step(kind, nid, u["old"]) if kind == "swap_op"
                         else dsl.step("replace", nid, u["old"]))
            if kind == "swap_op" and not dsl.snippet_ok(u["old"]):
                return None
        elif kind == "swap_args":
            hits = [n for n in nodes if n.type == "call" and n.start_byte == pos]
            if not hits:
                return None
            steps.append(dsl.step("swap_args", nodes.index(hits[0]), 0, 1))
        elif kind == "change_return":
            hits = [n for n in nodes if n.type == "return_statement" and n.start_byte == pos]
            if not hits:
                return None
            steps.append(dsl.step("change_return", nodes.index(hits[0]), u["old"].strip()))
        else:  # add_guard, insert_before, delete_stmt: statement starting at pos
            hits = [n for n in nodes if n.start_byte == pos and n.is_named
                    and n.parent is not None and n.parent.type in ("block", "module")]
            if not hits:
                return None
            nid = nodes.index(hits[0])
            if kind == "delete_stmt":
                steps.append(dsl.step("delete_stmt", nid))
            else:
                steps.append(dsl.step(kind, nid, u["old"].strip()))
    return steps


def payload_ok(step: str) -> bool:
    if step.startswith(("delete_stmt", "swap_args")):
        return True
    return dsl.snippet_ok(step[step.index(",") + 1:-1])


def build(count: int, seed: int):
    rng = random.Random(seed)
    fixtures = load_fixtures()
    pools = {}  # fixture -> mutation name -> proposals
    for name, fixed, _ in fixtures:
        _, nodes = parse(fixed)
        by_type = {}
        for p in proposals(fixed, nodes):
            by_type.setdefault(p.name, []).append(p)
        pools[name] = by_type
    seen, samples = set(), []
    tries = 0
    while len(samples) < count and tries < count * 50:
        tries += 1
        name, fixed, test = rng.choice(fixtures)
        types = pools[name]
        k = rng.choices([1, 2, 3], weights=[4, 4, 2])[0]
        chosen = []
        for _ in range(k):
            cand = rng.choice(types[rng.choice(sorted(types))])
            edits = [e for p in chosen for e in p.edits]
            if any(conflicts(cand.edits, [x]) for x in edits) or any(p is cand for p in chosen):
                continue
            chosen.append(cand)
        if not chosen:
            continue
        edits = [e for p in chosen for e in p.edits]
        buggy = apply_edits(fixed, edits)
        if buggy in seen or passes(buggy, test):
            continue
        steps = gold_steps(buggy, edits, chosen)
        if not steps or len(steps) != len(chosen) or len({s.split("(")[1][:3] for s in steps}) != len(steps):
            continue
        if not all(payload_ok(s) for s in steps):
            continue
        seen.add(buggy)
        samples.append({"fixture": name, "mutations": [p.name for p in chosen], "buggy": buggy,
                        "test": test, "fixed": fixed,
                        "gold": ";".join(steps) + "\n"})
    return samples


def split(samples, seed: int, holdout: float = 0.2):
    rng = random.Random(seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    cut = int(len(samples) * holdout)
    for rank, i in enumerate(order):
        samples[i]["split"] = "heldout" if rank < cut else "train"
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    samples = split(build(a.count, a.seed), a.seed)
    for i, s in enumerate(samples):
        s["id"] = i
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {len(samples)} samples to {a.out}")


if __name__ == "__main__":
    main()
