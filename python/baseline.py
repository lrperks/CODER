"""Whole-function rewrite baseline: the same model decoded in Free mode (no grammar, no forced tokens).

Usage: python -m python.baseline  (needs results/models/baseline.bin from policy.train --target baseline)
Writes results/baseline.json with dev pass@1 and pass@4 on the held-out split (the dev set).
"""
import json

from python.eval import RESULTS, rows, run_system


def main():
    res = run_system("baseline", rows(), workers=4)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
