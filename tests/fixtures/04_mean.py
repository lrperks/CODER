def f(xs):
    if not xs:
        return 0
    return sum(xs) // len(xs)
# --- test ---
def test_f():
    assert f([2, 4]) == 3
    assert f([]) == 0
    assert f([7, 8, 9, 10]) == 8
