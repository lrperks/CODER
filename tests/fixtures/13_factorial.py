def f(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
# --- test ---
def test_f():
    assert f(0) == 1
    assert f(4) == 24
    assert f(5) == 120
