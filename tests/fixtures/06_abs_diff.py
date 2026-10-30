def f(a, b):
    d = a - b
    if d < 0:
        d = -d
    return d
# --- test ---
def test_f():
    assert f(5, 2) == 3
    assert f(2, 5) == 3
    assert f(4, 4) == 0
