def f(deg_f):
    return (deg_f - 32) * 5 / 9
# --- test ---
def test_f():
    assert f(32) == 0
    assert f(212) == 100
    assert f(50) == 10
