import sympy as sp
from regelungstechnik import inverse_laplace, laplace_transform, partialbruchzerlegung


def test_laplace_transform_exponential():
    f = sp.exp(-2 * sp.symbols('t'))
    result = laplace_transform(f)
    assert sp.simplify(result['ergebnis'] - 1 / (sp.symbols('s') + 2)) == 0


def test_inverse_laplace_rational():
    t = sp.symbols('t')
    result = inverse_laplace(1 / (sp.symbols('s') + 3))
    expected = sp.exp(-3 * t) * sp.Heaviside(t)
    assert sp.simplify(result['ergebnis'] - expected) == 0


def test_partialbruch():
    result = partialbruchzerlegung([1, 2], [1, 3, 2])
    assert '1/(s + 1)' in str(result['ergebnis'])
