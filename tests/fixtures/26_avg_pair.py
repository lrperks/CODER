def f(a, b):
    return (a + b) / 2
# --- test ---
def test_f():
    assert f(2, 4) == 3
    assert f(0, 0) == 0
    assert f(-1, 1) == 0
