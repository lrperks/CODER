def f(x, lo, hi):
    return lo <= x and x <= hi
# --- test ---
def test_f():
    assert f(5, 0, 10) is True
    assert f(-1, 0, 10) is False
    assert f(10, 0, 10) is True
