def f(xs):
    out = []
    acc = 0
    for x in xs:
        acc += x
        out.append(acc)
    return out
# --- test ---
def test_f():
    assert f([1, 2, 3]) == [1, 3, 6]
    assert f([]) == []
    assert f([5, -5]) == [5, 0]
