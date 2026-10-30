def f(s, c):
    n = 0
    for ch in s:
        if ch == c:
            n += 1
    return n
# --- test ---
def test_f():
    assert f("banana", "a") == 3
    assert f("abc", "z") == 0
    assert f("", "a") == 0
