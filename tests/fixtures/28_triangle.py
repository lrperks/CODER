def f(n):
    return n * (n + 1) // 2
# --- test ---
def test_f():
    assert f(0) == 0
    assert f(4) == 10
    assert f(10) == 55
