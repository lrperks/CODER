def f(a, b):
    if a > b:
        return a
    return b
# --- test ---
def test_f():
    assert f(3, 2) == 3
    assert f(2, 5) == 5
    assert f(4, 4) == 4
