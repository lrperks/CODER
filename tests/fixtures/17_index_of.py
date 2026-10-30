def f(xs, v):
    for i in range(len(xs)):
        if xs[i] == v:
            return i
    return -1
# --- test ---
def test_f():
    assert f([5, 6, 7], 6) == 1
    assert f([5, 6, 7], 9) == -1
    assert f([], 1) == -1
