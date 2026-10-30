"""Verify throughput: fixed fixture sources through vedit.run_tests at 1, 4 and 8 workers.

Usage: python -m python.bench_verify   (prints JSON; median of 3 reps over 240 jobs)
"""
import json
import statistics
import sys
import time
from pathlib import Path

import vedit

from python.data.generate import load_fixtures

ROOT = Path(__file__).resolve().parents[1]
RUNNER = str(ROOT / "python" / "verify_runner.py")


def verify_throughput(workers_list=(1, 4, 8), reps=3, n_jobs=240):
    fixtures = load_fixtures()
    jobs = [(fixed, test) for _, fixed, test in fixtures]
    jobs = (jobs * (n_jobs // len(jobs) + 1))[:n_jobs]
    out = {}
    for w in workers_list:
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ok = vedit.run_tests(jobs, w, sys.executable, RUNNER)
            times.append(time.perf_counter() - t0)
            assert all(ok), "fixed fixture sources must pass"
        med = statistics.median(times)
        out[w] = {"tests_per_s": n_jobs / med, "wall_s": med}
    return out


if __name__ == "__main__":
    print(json.dumps(verify_throughput(), indent=2))
