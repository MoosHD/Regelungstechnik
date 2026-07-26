from __future__ import annotations

from pathlib import Path

import sympy as sp

import regelungstechnik as rt


def test_all_exports_present():
    for name in rt.__all__:
        assert hasattr(rt, name), f"Export fehlt: {name}"


def test_laplace_functions_smoke():
    t = sp.symbols("t", positive=True)
    f_t = t * sp.exp(-2 * t)
    res_l = rt.laplace_transform(f_t)
    assert "ergebnis" in res_l

    res_il = rt.inverse_laplace(res_l["ergebnis"])
    assert "ergebnis" in res_il

    res_pb = rt.partialbruchzerlegung([1, 3], [1, 4, 3])
    assert "ergebnis" in res_pb


def test_stability_functions_smoke():
    den = [1, 3, 2]
    num = [1]

    res_h = rt.hurwitz_kriterium(den)
    res_r = rt.routh_kriterium(den)
    res_n = rt.nyquist_kriterium(num, den)

    assert isinstance(res_h["ergebnis"]["stabil"], bool)
    assert isinstance(res_r["ergebnis"]["stabil"], bool)
    assert isinstance(res_n["ergebnis"]["stabil"], bool)


def test_connection_and_design_functions_smoke():
    g1 = ([1], [1, 1])
    g2 = ([2], [1, 2])

    res_reihe = rt.reihenschaltung(g1, g2)
    res_parallel = rt.parallelschaltung(g1, g2)
    res_rk = rt.rueckkopplung(res_reihe["ergebnis"], ([1], [1, 5]), negativ=True)

    assert len(res_reihe["ergebnis"]) == 2
    assert len(res_parallel["ergebnis"]) == 2
    assert len(res_rk["ergebnis"]) == 2

    res_reg = rt.reglerparameter_nach_verfahren(
        reglertyp="PI",
        verfahren="ziegler-nichols",
        modus="offen",
        K=2.0,
        T=5.0,
        K_T=1.0,
    )
    res_phase = rt.phasenkorrekturglied_auslegung("anhebend", phi_grad=35.0, omega_c=2.0, K=1.0)
    res_w = rt.wurzelortsauslegung([1], [1, 3, 2], k_start=0.0, k_ende=8.0, anzahl_k=41)

    assert "Kp" in res_reg["ergebnis"]
    assert "num" in res_phase["ergebnis"]
    assert len(res_w["ergebnis"]["k_empfohlen"]) >= 1


def test_time_response_metrics_smoke():
    num = [1]
    den = [1, 3, 2]

    res_fex = rt.sprungantwort_mit_fex((num, den), F_ex=([1], [1, 1]), t_ende=4.0)
    res_sf = rt.sprungfaehigkeit_realisierbarkeit(num, den)
    res_sa = rt.stationaere_abweichung(num, den)
    res_mu = rt.maximale_ueberschwingweite([1], [1, 2, 1])
    res_az = rt.ausregelzeit([1], [1, 2, 1], toleranzband=0.05)

    assert len(res_fex["ergebnis"][0]) > 2
    assert "realisierbar" in res_sf["ergebnis"]
    assert "abweichung" in res_sa["ergebnis"]
    assert "ueberschwingweite" in res_mu["ergebnis"]
    assert "ausregelzeit" in res_az["ergebnis"]


def test_state_space_functions_smoke():
    a = [[0, 1], [-2, -3]]
    b = [[0], [1]]
    c = [[1, 0]]
    d = [[0]]

    res_tf = rt.zustandsraum_zu_uebertragungsfunktion(a, b, c, d)
    num_tf, den_tf = res_tf["ergebnis"]

    res_rnf = rt.regelungsnormalform(num_tf, den_tf)
    res_tm = rt.transitionsmatrix(a, [0.0, 0.5, 1.0])
    res_tms = rt.transitionsmatrix_symbolisch(a)
    res_j = rt.jordan_normalform(a)
    res_pk = rt.poincare_klassifikation(a)

    assert len(res_rnf["ergebnis"]) == 4
    assert len(res_tm["ergebnis"]) == 3
    assert res_tms["ergebnis"] is not None
    assert "J" in res_j["ergebnis"]
    assert "typ" in res_pk["ergebnis"]


def test_plot_functions_smoke():
    num = [1]
    den = [1, 3, 2]

    plots = [
        rt.plot_sprungantwort(num, den, dateiname="smoke_sprung.png"),
        rt.plot_impulsantwort(num, den, dateiname="smoke_impuls.png"),
        rt.plot_bode(num, den, dateiname="smoke_bode.png"),
        rt.plot_ortskurve(num, den, dateiname="smoke_ort.png"),
        rt.plot_nyquist(num, den, dateiname="smoke_nyquist.png"),
        rt.plot_pol_nullstellen(num, den, dateiname="smoke_polnull.png"),
        rt.plot_wurzelortskurve(num, den, dateiname="smoke_wurzelort.png"),
    ]

    for item in plots:
        pfad = item.get("plot_pfad")
        assert pfad
        assert Path(pfad).exists()

    p_path = rt.plot_poincare([[0, 1], [-2, -3]], dateiname="smoke_poincare.png")
    assert Path(p_path).exists()


def test_cli_export_present():
    assert rt.cli is not None
