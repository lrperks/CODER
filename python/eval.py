"""Held-out evaluation: pass@1/4, token and verify costs, forced-token share, decode and verify speed.

Usage: python -m python.eval  (writes results/eval.json)
"""
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import vedit
from python.data.generate import FIXTURES  # noqa: F401  (keeps the dataset module importable)
from python.policy import reference
from python.policy import check

ROOT = Path(__file__).resolve().parents[2]
RUNNER = str(ROOT / "python" / "verify_runner.py")
PY = "/usr/bin/python3.11"
RESULTS = ROOT / "results"


def rows():
    out = [json.loads(line) for line in (ROOT / "data" / "samples.jsonl").read_text().splitlines()]
    return [r for r in out if r["split"] == "heldout"]


def run_system(name, hold, workers, avx2=True):
    model = vedit.Model(str(RESULTS / "models" / f"{name}.bin"))
    edit = name == "edit"
    prompts = [vedit.render_prompt(r["test"], r["buggy"], edit) for r in hold]
    samples = [(p, r["buggy"], r["test"]) for p, r in zip(prompts, hold)]
    t0 = time.perf_counter()
    res = vedit.solve(model, samples, vedit.Mode.Edit if edit else vedit.Mode.Free,
                      4, 160 if edit else 400, workers, 0, avx2, PY, RUNNER)
    wall = time.perf_counter() - t0
    p1 = sum(r[0] for r in res) / len(res)
    p4 = sum(r[1] for r in res) / len(res)
    wins = [r for r in res if r[1]]
    toks = sum(r[2] for r in res)
    forced = sum(r[3] for r in res)
    return {
        "n": len(res), "pass1": p1, "pass4": p4, "wall_s": wall,
        "mean_output_tokens_per_success": statistics.mean(r[2] for r in wins) if wins else None,
        "mean_verify_calls_per_success": statistics.mean(r[4] for r in wins) if wins else None,
        "forced_fraction": forced / toks if toks else 0.0,
        "total_tokens": toks, "forced_tokens": forced,
    }


def decode_speed(name, runs=3, n_tokens=200):
    """Fixed prompt, n_tokens generated with no grammar and EOS ignored; median of runs."""
    prompt = check.heldout_prompts(name, 1)[0]
    cpp = vedit.Model(str(RESULTS / "models" / f"{name}.bin"))
    ref = reference.load_reference(name)
    c_times, p_times = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        vedit.decode(cpp, prompt.decode(), vedit.Mode.Free, n_tokens, 0.0, 0, False, True)
        c_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        reference.decode(ref, prompt, False, n_tokens, stop_at_eos=False)
        p_times.append(time.perf_counter() - t0)
    cpp_tps = n_tokens / statistics.median(c_times)
    py_tps = n_tokens / statistics.median(p_times)
    return {"cpp_tok_s": cpp_tps, "pytorch_tok_s": py_tps, "ratio": cpp_tps / py_tps,
            "cpp_runs_s": c_times, "pytorch_runs_s": p_times}


def verify_speed(hold, workers_list=(1, 4, 8), reps=3):
    jobs = [(r["fixed"], r["test"]) for r in hold[:240]]
    out = {}
    for w in workers_list:
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ok = vedit.run_tests(jobs, w, PY, RUNNER)
            times.append(time.perf_counter() - t0)
            assert all(ok), "fixed sources must pass"
        out[w] = {"tests_per_s": len(jobs) / statistics.median(times), "wall_s": statistics.median(times)}
    return out


def main():
    hold = rows()
    bad_edit, _ = check.parity("edit")
    bad_base, _ = check.parity("baseline")
    cpp_valid, re_valid = check.grammar_validity("edit", n=len(hold))
    result = {
        "parity": {"edit_mismatches": len(bad_edit), "baseline_mismatches": len(bad_base),
                   "edit_first_mismatch": bad_edit[:1], "baseline_first_mismatch": bad_base[:1]},
        "grammar_validity": {"cpp": cpp_valid, "python_re": re_valid},
        "edit": run_system("edit", hold, workers=4),
        "baseline": run_system("baseline", hold, workers=4),
        "decode_speed": decode_speed("edit"),
        "verify_speed": verify_speed(hold),
        "platform": platform.platform(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
