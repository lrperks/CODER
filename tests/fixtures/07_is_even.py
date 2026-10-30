def f(n):
    return n % 2 == 0
# --- test ---
def test_f():
    assert f(4) is True
    assert f(7) is False
    assert f(0) is True
