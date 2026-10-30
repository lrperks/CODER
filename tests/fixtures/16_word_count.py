def f(s):
    if not s:
        return 0
    return len(s.split())
# --- test ---
def test_f():
    assert f("a b  c") == 3
    assert f("") == 0
    assert f("one") == 1
