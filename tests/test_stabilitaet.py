from regelungstechnik import hurwitz_kriterium, routh_kriterium


def test_hurwitz_simple_stable():
    result = hurwitz_kriterium([1, 5, 6])
    assert result['ergebnis']['stabil'] is True


def test_routh_simple_stable():
    result = routh_kriterium([1, 5, 6])
    assert result['ergebnis']['stabil'] is True
