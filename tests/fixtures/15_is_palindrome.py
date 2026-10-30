def f(s):
    if s is None: return False
    return s == s[::-1]
# --- test ---
def test_f():
    assert f("abba") is True
    assert f("abc") is False
    assert f(None) is False
