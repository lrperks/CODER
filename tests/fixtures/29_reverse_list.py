def f(xs):
    out = []
    for x in reversed(xs):
        out.append(x)
    return out
# --- test ---
def test_f():
    assert f([1, 2, 3]) == [3, 2, 1]
    assert f([]) == []
    assert f([7]) == [7]
