import pytest

from python.policy import check

MODELS = check.ROOT / "results" / "models"


@pytest.mark.parametrize("name", ["edit", "baseline"])
def test_cpp_runtime_matches_pytorch_reference(name):
    if not (MODELS / f"{name}.bin").exists():
        pytest.skip(f"train {name} first")
    bad, total = check.parity(name, n=100)
    assert total == 100
    assert not bad, f"{len(bad)}/{total} greedy outputs differ, first: {bad[0]}"


def test_edit_policy_outputs_are_grammar_valid():
    if not (MODELS / "edit.bin").exists():
        pytest.skip("train edit first")
    cpp_frac, re_frac = check.grammar_validity("edit", n=100)
    assert cpp_frac == 1.0 and re_frac == 1.0
