def f(xs):
    count = 0
    for x in xs:
        if x > 0:
            count += 1
    return count
# --- test ---
def test_f():
    assert f([1, -2, 3]) == 2
    assert f([]) == 0
    assert f([-1, -1]) == 0
