def f(xs, cap):
    total = 0
    for x in xs:
        if total + x > cap:
            break
        total += x
    return total
# --- test ---
def test_f():
    assert f([1, 2, 3, 4], 6) == 6
    assert f([5, 5], 3) == 0
    assert f([], 3) == 0
