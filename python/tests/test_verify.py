import sys
from pathlib import Path

import pytest
import vedit

from python.data.generate import load_fixtures

ROOT = Path(__file__).resolve().parents[2]
RUNNER = str(ROOT / "python" / "verify_runner.py")
MODEL = ROOT / "results" / "models" / "edit.bin"


def test_run_tests_pass_fail_and_order_at_any_worker_count():
    fixtures = load_fixtures()
    fixed = [(src, test) for _, src, test in fixtures]
    broken = [("def f(*args):\n    raise ValueError('broken')\n", test) for _, test in fixed]
    jobs = [j for pair in zip(fixed, broken) for j in pair]
    expect = [True, False] * len(fixed)
    for w in (1, 4, 8):
        assert vedit.run_tests(jobs, w, sys.executable, RUNNER) == expect, f"workers={w}"


def test_fixed_sources_never_time_out_under_concurrency():
    fixed = [(src, test) for _, src, test in load_fixtures()]
    jobs = (fixed * 10)[:300]
    for w in (1, 4, 8):
        assert vedit.run_tests(jobs, w, sys.executable, RUNNER) == [True] * 300, f"workers={w}"


def test_solve_reports_six_fields_per_sample():
    if not MODEL.exists():
        pytest.skip("train edit first")
    model = vedit.Model(str(MODEL))
    row = (vedit.render_prompt("def test_f():\n    assert f(1) == 1\n", "def f(x):\n    return 0\n", True),
           "def f(x):\n    return 0\n", "def test_f():\n    assert f(1) == 1\n")
    res = vedit.solve(model, [row], vedit.Mode.Edit, 4, 160, 1, 0, True, sys.executable, RUNNER)
    pass1, pass4, tokens, forced, calls, wall_s = res[0]
    assert 0 <= calls <= 4 and wall_s > 0 and forced <= tokens
