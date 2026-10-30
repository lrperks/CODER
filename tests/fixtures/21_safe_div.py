def f(a, b):
    if b is None: return None
    if b == 0:
        return None
    return a / b
# --- test ---
def test_f():
    assert f(6, 3) == 2
    assert f(1, 0) is None
    assert f(1, None) is None
