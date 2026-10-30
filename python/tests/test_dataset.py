import json

from python.data.generate import FIXTURES, build, load_fixtures, passes, split, OUT

MUTATIONS = {"off_by_one", "wrong_arith_op", "wrong_compare_op", "swapped_args", "wrong_return",
             "missing_none_guard", "deleted_stmt", "early_return", "negate_condition", "wrong_var"}


def test_fixtures_are_correct():
    fixtures = load_fixtures()
    assert len(fixtures) == 30
    for name, fixed, test in fixtures:
        assert passes(fixed, test), name


def test_mutations_fail_their_tests():
    samples = build(1000, seed=1)
    assert len(samples) >= 1000
    for s in samples:
        assert not passes(s["buggy"], s["test"]), s["fixture"]
        assert passes(s["fixed"], s["test"]), s["fixture"]


def test_all_mutation_types_present():
    samples = build(1000, seed=1)
    used = {m for s in samples for m in s["mutations"]}
    assert used == MUTATIONS


def test_seeded_and_split():
    a = split(build(200, seed=2), seed=2)
    b = split(build(200, seed=2), seed=2)
    assert a == b
    heldout = sum(s["split"] == "heldout" for s in a)
    assert abs(heldout / len(a) - 0.2) < 0.01


def test_committed_dataset_matches_spec():
    if not OUT.exists():
        return
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    assert len(rows) >= 1000
    assert sum(r["split"] == "heldout" for r in rows) / len(rows) == 0.2
