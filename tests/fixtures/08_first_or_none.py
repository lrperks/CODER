def f(xs):
    if xs is None: return None
    if len(xs) == 0:
        return None
    return xs[0]
# --- test ---
def test_f():
    assert f([9, 1]) == 9
    assert f([]) is None
    assert f(None) is None
