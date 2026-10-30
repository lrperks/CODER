def f(xs):
    best = None
    for x in xs:
        if best is None or x > best:
            best = x
    return best
# --- test ---
def test_f():
    assert f([3, 1, 4, 1, 5]) == 5
    assert f([-2, -7]) == -2
    assert f([]) is None
