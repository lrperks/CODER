def f(xs):
    if xs is None: return None
    total = 0
    for x in xs:
        total += x
    return total
# --- test ---
def test_f():
    assert f([1, 2, 3]) == 6
    assert f([]) == 0
    assert f(None) is None
