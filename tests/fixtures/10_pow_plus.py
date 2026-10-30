def f(a, b):
    return pow(a, 2) + b
# --- test ---
def test_f():
    assert f(3, 1) == 10
    assert f(0, 5) == 5
    assert f(-2, 0) == 4
