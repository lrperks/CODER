def f(a, b):
    while b != 0:
        a, b = b, a % b
    return a
# --- test ---
def test_f():
    assert f(12, 18) == 6
    assert f(7, 3) == 1
    assert f(9, 0) == 9
