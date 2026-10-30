"""Decode speed: C++ runtime vs PyTorch reference, fixed prompt, 200 tokens, 3 runs, median.

Usage: python -m python.bench_decode [--name edit]
Without --name, a randomly initialised model (seed 0) is exported and benchmarked; decode
speed depends only on the architecture, not on the weight values.
Writes results/decode_speed.json.
"""
import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

import torch

import vedit
from python.policy import reference
from python.policy.export import dequantized, export
from python.policy.model import CFG, CharLM

ROOT = Path(__file__).resolve().parents[1]
PROMPT = vedit.render_prompt("def f(x):\n    return x + 1\n", "def f(x):\n    return x - 1\n", False).encode()
N_TOKENS, RUNS = 200, 3


def load(name):
    if name:
        return vedit.Model(str(ROOT / "results" / "models" / f"{name}.bin")), reference.load_reference(name)
    torch.manual_seed(0)
    m = CharLM(CFG)
    path = Path(tempfile.mkdtemp()) / "random.bin"
    export(m, path)
    return vedit.Model(str(path)), dequantized(m).eval()


def cpu_model():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def timed(fn):
    runs = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - t0)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="trained model name in results/models (default: random)")
    args = ap.parse_args()
    cpp, ref = load(args.name)
    cpp_runs = timed(lambda: vedit.decode(cpp, PROMPT.decode(), vedit.Mode.Free, N_TOKENS, 0.0, 0, False, True))
    py_runs = timed(lambda: reference.decode(ref, PROMPT, False, N_TOKENS, stop_at_eos=False))
    cpp_tps = N_TOKENS / statistics.median(cpp_runs)
    py_tps = N_TOKENS / statistics.median(py_runs)
    out = {
        "model": args.name or "random-init-seed0",
        "tokens": N_TOKENS, "runs": RUNS, "prompt_bytes": len(PROMPT),
        "cpu": cpu_model(), "torch": torch.__version__, "python": platform.python_version(),
        "cxx_flags": "-O3 -march=native -DNDEBUG",  # CMakeLists.txt, Release
        "cpp_tok_s": cpp_tps, "pytorch_tok_s": py_tps, "ratio": cpp_tps / py_tps,
        "cpp_runs_s": cpp_runs, "pytorch_runs_s": py_runs,
    }
    path = ROOT / "results" / "decode_speed.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
