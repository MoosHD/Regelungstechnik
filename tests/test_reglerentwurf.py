import sympy as sp
from regelungstechnik import (parallelschaltung, reihenschaltung, rueckkopplung,
                              sprungantwort_mit_fex, sprungfaehigkeit_realisierbarkeit,
                              stationaere_abweichung, maximale_ueberschwingweite,
                              ausregelzeit, reglerparameter_nach_verfahren,
                              wurzelortsauslegung, phasenkorrekturglied_auslegung)


def _parse_transfer_function_expression(text: str):
    expr = sp.sympify(text, evaluate=True)
    s = sp.symbols('s')
    num, den = sp.fraction(sp.together(expr))
    num_poly = sp.Poly(sp.expand(num), s)
    den_poly = sp.Poly(sp.expand(den), s)

    def _normalize(coeffs):
        result = []
        for c in coeffs:
            c = sp.simplify(c)
            if c.free_symbols:
                result.append(c)
            else:
                result.append(float(sp.nsimplify(c)))
        return result

    return _normalize(num_poly.all_coeffs()), _normalize(den_poly.all_coeffs())


def test_reihenschaltung():
    result = reihenschaltung(([1], [1, 1]), ([1], [1, 2]))
    assert result['ergebnis'][1] == [1, 3, 2]


def test_parallelschaltung():
    result = parallelschaltung(([1], [1, 1]), ([1], [1, 2]))
    assert result['ergebnis'][1] == [1, 3, 2]


def test_sprungfaehigkeit():
    result = sprungfaehigkeit_realisierbarkeit([1, 2], [1, 3, 2])
    assert result['ergebnis']['sprungfaehig'] is False


def test_sprungantwort_mit_fex():
    result = sprungantwort_mit_fex(([1], [1, 1]), ([1], [1, 0]), t_ende=2.0)
    assert len(result['ergebnis'][0]) > 1
    assert result['ergebnis'][1][0] == 0.0


def test_parse_transfer_function_expression():
    num, den = _parse_transfer_function_expression("1/(s+1)")
    assert num == [1.0]
    assert den == [1.0, 1.0]


def test_sprungantwort_symbolic_placeholders():
    result = sprungantwort_mit_fex((["C"], ["A", "B", "-1"]), (["D"], [1]))
    assert "symbolisch" in result['ergebnis']['status'].lower()

def test_stationaere_abweichung():
    result = stationaere_abweichung([1], [1, 1])
    assert abs(result['ergebnis']['abweichung'] - 0.0) < 1e-8


def test_maximale_ueberschwingweite():
    result = maximale_ueberschwingweite([1], [1, 2, 1], t_ende=10.0)
    assert 0.0 <= result['ergebnis']['ueberschwingweite'] < 1.0


def test_ausregelzeit():
    result = ausregelzeit([1], [1, 2, 1], toleranzband=0.05, t_ende=10.0)
    assert result['ergebnis']['ausregelzeit'] >= 0.0

def test_parse_transfer_function_expression_with_placeholders():
    num, den = _parse_transfer_function_expression('C/((A*s^2)+(B*s)-1)')
    assert num == [sp.Symbol('C')]
    assert den == [sp.Symbol('A'), sp.Symbol('B'), -1]


def test_reglerparameter_zn_offen_pid():
    result = reglerparameter_nach_verfahren(
        reglertyp="PID",
        verfahren="ziegler-nichols",
        modus="offen",
        K=2.0,
        T=5.0,
        K_T=1.0,
    )
    assert result["ergebnis"]["verfahren"].startswith("Ziegler-Nichols")
    assert result["ergebnis"]["Kp"] > 0.0
    assert result["ergebnis"]["Ti"] is not None


def test_reglerparameter_cc_gueltigkeit_warnung():
    result = reglerparameter_nach_verfahren(
        reglertyp="PI",
        verfahren="cohen-coon",
        K=1.0,
        T=1.0,
        K_T=2.5,
    )
    assert result["ergebnis"]["gueltig"] is False
    assert any("Achtung" in hint for hint in result.get("hinweise", []))


def test_wurzelortsauslegung_basic():
    result = wurzelortsauslegung([1.0], [1.0, 3.0, 2.0], k_start=0.0, k_ende=10.0, anzahl_k=31)
    ergebnis = result["ergebnis"]
    assert len(ergebnis["k_empfohlen"]) >= 1
    assert len(ergebnis["pole_tabelle"]) >= 1
    assert "asymptoten" in ergebnis


def test_phasenkorrektur_anhebend_basic():
    result = phasenkorrekturglied_auslegung("anhebend", phi_grad=35.0, omega_c=2.0, K=1.0)
    ergebnis = result["ergebnis"]
    assert ergebnis["typ"] == "anhebend"
    assert ergebnis["phi_grad"] > 0.0
    assert len(ergebnis["num"]) == 2
    assert len(ergebnis["den"]) == 2


def test_phasenkorrektur_absenkend_basic():
    result = phasenkorrekturglied_auslegung("absenkend", phi_grad=20.0, omega_c=1.5, K=2.0)
    ergebnis = result["ergebnis"]
    assert ergebnis["typ"] == "absenkend"
    assert ergebnis["phi_grad"] < 0.0
    assert ergebnis["K"] == 2.0
