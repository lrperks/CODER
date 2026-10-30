def f(s):
    out = ""
    for ch in s:
        if ch not in "aeiou":
            out += ch
    return out
# --- test ---
def test_f():
    assert f("banana") == "bnn"
    assert f("xyz") == "xyz"
    assert f("") == ""
