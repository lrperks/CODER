def f(xs):
    for x in xs:
        if x < 0:
            return True
    return False
# --- test ---
def test_f():
    assert f([1, -1]) is True
    assert f([1, 2]) is False
    assert f([]) is False
