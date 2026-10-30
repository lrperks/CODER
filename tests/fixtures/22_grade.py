def f(score):
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    return "C"
# --- test ---
def test_f():
    assert f(95) == "A"
    assert f(85) == "B"
    assert f(10) == "C"
