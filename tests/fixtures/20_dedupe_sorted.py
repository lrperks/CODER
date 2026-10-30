def f(xs):
    out = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return out
# --- test ---
def test_f():
    assert f([1, 1, 2, 3, 3]) == [1, 2, 3]
    assert f([]) == []
    assert f([4]) == [4]
