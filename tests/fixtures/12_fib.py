def f(n):
    a = 0
    b = 1
    for _ in range(n):
        a, b = b, a + b
    return a
# --- test ---
def test_f():
    assert f(0) == 0
    assert f(1) == 1
    assert f(10) == 55
