def f(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
# --- test ---
def test_f():
    assert f(5, 0, 10) == 5
    assert f(-3, 0, 10) == 0
    assert f(12, 0, 10) == 10
