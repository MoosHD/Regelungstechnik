from regelungstechnik import hurwitz_kriterium, nyquist_kriterium, routh_kriterium


def test_hurwitz_simple_stable():
    result = hurwitz_kriterium([1, 5, 6])
    assert result['ergebnis']['stabil'] is True


def test_routh_simple_stable():
    result = routh_kriterium([1, 5, 6])
    assert result['ergebnis']['stabil'] is True


def test_nyquist_simple_stable_closed_loop():
    # L(s)=1/(s+1) => 1+L(s) = (s+2)/(s+1), stabiler geschlossener Kreis
    result = nyquist_kriterium([1], [1, 1], w_min=1e-3, w_max=1e3, punkte=2000)
    assert result['ergebnis']['stabil'] is True
    assert result['ergebnis']['Z'] == 0


def test_nyquist_accepts_factorized_transfer_function_string():
    coeff_result = nyquist_kriterium([5], [1, 5, 3, -9], w_min=1e-3, w_max=1e3, punkte=2000)
    expr_result = nyquist_kriterium(
        'k_R/((s+3)^2(s-1))',
        w_min=1e-3,
        w_max=1e3,
        punkte=2000,
        substitutions={'k_R': 5},
    )
    assert expr_result['ergebnis']['P'] == coeff_result['ergebnis']['P']
    assert expr_result['ergebnis']['N_cw'] == coeff_result['ergebnis']['N_cw']
    assert expr_result['ergebnis']['Z'] == coeff_result['ergebnis']['Z']
