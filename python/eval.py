"""Held-out evaluation: pass@1/4, token and verify costs, forced-token share, decode and verify speed.

Usage (from repo root, after building the module into ./build): python -m python.eval
Writes results/eval.json (raw), results/config.json and results/report.md.
"""
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

import vedit
from python.policy import check, reference
from python.policy import check
from python.policy.model import CFG
from python.bench_verify import verify_throughput
from python.verify_runner import MEM_LIMIT

ROOT = Path(__file__).resolve().parents[1]
RUNNER = str(ROOT / "python" / "verify_runner.py")
PY = sys.executable
RESULTS = ROOT / "results"
K = 4
# Baseline budget fills the 640-token context after the longest held-out prompt (267 chars).
MAX_NEW = {"edit": 160, "baseline": check.max_new_tokens("baseline")}


def rows():
    out = [json.loads(line) for line in (ROOT / "data" / "samples.jsonl").read_text().splitlines()]
    return [r for r in out if r["split"] == "heldout"]


def run_system(name, hold, workers):
    model = vedit.Model(str(RESULTS / "models" / f"{name}.bin"))
    edit = name == "edit"
    samples = [(vedit.render_prompt(r["test"], r["buggy"], edit), r["buggy"], r["test"]) for r in hold]
    t0 = time.perf_counter()
    res = vedit.solve(model, samples, vedit.Mode.Edit if edit else vedit.Mode.Free,
                      K, MAX_NEW[name], workers, 0, True, PY, RUNNER)
    wall = time.perf_counter() - t0
    wins = [r for r in res if r[1]]
    toks = sum(r[2] for r in res)
    forced = sum(r[3] for r in res)
    return {
        "n": len(res), "pass1": sum(r[0] for r in res) / len(res), "pass4": sum(r[1] for r in res) / len(res),
        "wall_s": wall,
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


def config():
    cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
               if l.startswith("model name"))
    flags = next(l.split("=", 1)[1].strip() for l in (ROOT / "build" / "CMakeFiles" / "vedit.dir" / "flags.make")
                 .read_text().splitlines() if l.startswith("CXX_FLAGS"))
    compiler = subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {
        "git_commit": commit,
        "model": CFG,
        "seeds": {"dataset": 0, "eval_prompts": 0, "solve_sampling": 0},
        "k": K, "max_new_tokens": MAX_NEW, "decode_tokens": 200, "decode_runs": 3,
        "verify_timeout_s": 2.0, "verify_mem_limit_bytes": MEM_LIMIT, "verify_workers": [1, 4, 8],
        "compiler": compiler, "cxx_flags": flags, "cpu": cpu, "cpu_count": subprocess.run(
            ["nproc"], capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
    }


def criteria(r):
    e, b = r["edit"], r["baseline"]
    if e["mean_output_tokens_per_success"] and b["mean_output_tokens_per_success"]:
        ratio = e["mean_output_tokens_per_success"] / b["mean_output_tokens_per_success"]
    else:
        ratio = None
    par = r["parity"]["edit_mismatches"] + r["parity"]["baseline_mismatches"]
    return [
        ("Policy outputs are 100% grammar-valid", r["grammar_validity"]["cpp"] == 1.0,
         f"{r['grammar_validity']['cpp']:.1%} (C++), {r['grammar_validity']['python_re']:.1%} (Python re)"),
        ("C++ runtime matches PyTorch reference (100 prompts)", par == 0,
         f"{par} mismatches"),
        ("Edit pass@4 >= baseline pass@4", e["pass4"] >= b["pass4"],
         f"{e['pass4']:.1%} vs {b['pass4']:.1%}"),
        ("Edit uses <= 1/3 of baseline output tokens per successful fix",
         ratio is not None and ratio <= 1 / 3,
         f"{ratio:.2f}x" if ratio is not None else "no successes"),
        (">= 30% of emitted tokens from forced-token shortcuts", e["forced_fraction"] >= 0.30,
         f"{e['forced_fraction']:.1%}"),
        ("C++ decode tok/s >= 5x PyTorch reference", r["decode_speed"]["ratio"] >= 5,
         f"{r['decode_speed']['ratio']:.2f}x"),
    ]


def pct(x):
    return f"{x:.1%}"


def report(r):
    e, b = r["edit"], r["baseline"]
    d, v = r["decode_speed"], r["verify_speed"]
    p, g = r["parity"], r["grammar_validity"]
    fmt = lambda x: "n/a" if x is None else f"{x:.1f}"
    lines = [
        "# Eval report", "",
        f"Held-out split: {e['n']} samples. Raw numbers: `results/eval.json`. Config: `results/config.json`.", "",
        "## Systems", "",
        "| metric | edit (grammar + forced tokens) | baseline (whole file) |", "|---|---|---|",
        f"| pass@1 | {pct(e['pass1'])} | {pct(b['pass1'])} |",
        f"| pass@4 | {pct(e['pass4'])} | {pct(b['pass4'])} |",
        f"| mean output chars per successful fix | {fmt(e['mean_output_tokens_per_success'])} | "
        f"{fmt(b['mean_output_tokens_per_success'])} |",
        f"| mean verify calls per successful fix | {fmt(e['mean_verify_calls_per_success'])} | "
        f"{fmt(b['mean_verify_calls_per_success'])} |",
        f"| forced-token fraction | {pct(e['forced_fraction'])} | 0 (no grammar) |",
        f"| eval wall time (s) | {e['wall_s']:.1f} | {b['wall_s']:.1f} |", "",
        "## Decode speed (fixed prompt, 200 tokens, median of 3, CPU)", "",
        "| system | tok/s | runs (s) |", "|---|---|---|",
        f"| C++ runtime | {d['cpp_tok_s']:.1f} | {', '.join(f'{t:.3f}' for t in d['cpp_runs_s'])} |",
        f"| PyTorch reference | {d['pytorch_tok_s']:.1f} | {', '.join(f'{t:.3f}' for t in d['pytorch_runs_s'])} |",
        f"| ratio | {d['ratio']:.2f}x | |", "",
        "## Verify throughput (240 fixed programs, median of 3)", "",
        "| workers | tests/s | wall (s) |", "|---|---|---|",
        *[f"| {w} | {x['tests_per_s']:.1f} | {x['wall_s']:.2f} |" for w, x in v.items()], "",
        "## Correctness checks", "",
        f"- Greedy parity, edit: {p['edit_mismatches']} of 100 mismatches.",
        f"- Greedy parity, baseline: {p['baseline_mismatches']} of 100 mismatches.",
        f"- Grammar validity: C++ {pct(g['cpp'])}, Python `re` {pct(g['python_re'])}.",
        "- Scalar vs AVX2 matmul agreement: not measured by this script; see `cpp/tests/test_matmul.cpp`.", "",
        "## Success criteria", "", "| criterion | met | measured |", "|---|---|---|",
        *[f"| {name} | {'yes' if ok else '**no**'} | {val} |" for name, ok, val in criteria(r)], "",
        "## Failures", "",
    ]
    failed = [name for name, ok, _ in criteria(r) if not ok]
    lines += [f"- Missed: {name}" for name in failed] or ["- None of the success criteria were missed."]
    for side in ("edit", "baseline"):
        for m in r["parity_first"][side]:
            lines.append(f"- Parity mismatch ({side}, prompt {m['prompt']}): C++ `{m['cpp']!r}` vs ref `{m['ref']!r}`")
    return "\n".join(lines) + "\n"


def main():
    hold = rows()
    bad_edit, _ = check.parity("edit")
    bad_base, _ = check.parity("baseline")
    cpp_valid, re_valid = check.grammar_validity("edit", n=100)
    result = {
        "parity": {"edit_mismatches": len(bad_edit), "baseline_mismatches": len(bad_base)},
        "parity_first": {"edit": bad_edit[:5], "baseline": bad_base[:5]},
        "grammar_validity": {"cpp": cpp_valid, "python_re": re_valid},
        "edit": run_system("edit", hold, workers=4),
        "baseline": run_system("baseline", hold, workers=4),
        "decode_speed": decode_speed("edit"),
        "verify_speed": verify_throughput(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(result, indent=2))
    (RESULTS / "config.json").write_text(json.dumps(config(), indent=2))
    (RESULTS / "report.md").write_text(report(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
