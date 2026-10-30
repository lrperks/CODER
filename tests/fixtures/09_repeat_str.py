def f(s, n):
    out = ""
    for i in range(n):
        out = out + s
    return out
# --- test ---
def test_f():
    assert f("ab", 3) == "ababab"
    assert f("x", 0) == ""
    assert f("", 4) == ""
