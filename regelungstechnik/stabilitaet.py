"""Stabilitaetskriterien und Routh-Hurwitz-Berechnungen."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy import signal
import sympy as sp
from sympy.parsing.sympy_parser import (auto_symbol, convert_xor,
                                        implicit_multiplication_application,
                                        parse_expr, standard_transformations)

EPSILON = 1e-12
SYMPY_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    auto_symbol,
)


def _parse_symbolic_expression(text: str,
                               substitutions: Mapping[str, float] | None = None) -> sp.Expr:
    local_dict: dict[str, Any] = {'s': sp.Symbol('s')}
    if substitutions:
        local_dict.update(substitutions)
    try:
        return parse_expr(text, local_dict=local_dict, transformations=SYMPY_TRANSFORMATIONS)
    except Exception as exc:  # pragma: no cover - sympy errors are format-dependent
        raise ValueError(f"Ungueltige Funktions-Eingabe: {text}") from exc


def _coerce_polynomial_coeffs(poly: Sequence[float] | np.ndarray | str,
                              substitutions: Mapping[str, float] | None = None,
                              name: str = 'Polynom') -> np.ndarray:
    if isinstance(poly, str):
        expr = _parse_symbolic_expression(poly, substitutions=substitutions)
        poly_expr = sp.expand(expr)
        symbolic_coeffs = sp.Poly(poly_expr, sp.Symbol('s')).all_coeffs()
        coeffs: list[float] = []
        for coeff in symbolic_coeffs:
            unresolved = coeff.free_symbols - {sp.Symbol('s')}
            if unresolved:
                namen = ', '.join(sorted(str(symbol) for symbol in unresolved))
                raise ValueError(f"{name} enthaelt unaufgeloeste Parameter: {namen}")
            coeff_eval = sp.N(coeff)
            if coeff_eval.is_real is False:
                raise ValueError(f"{name} muss reelle Koeffizienten haben.")
            coeffs.append(float(coeff_eval))
        return np.array(coeffs, dtype=float)

    return np.array(poly, dtype=float)


def parse_uebertragungsfunktion(num: Sequence[float] | np.ndarray | str,
                                den: Sequence[float] | np.ndarray | str | None = None,
                                substitutions: Mapping[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if den is None:
        if not isinstance(num, str):
            raise ValueError("Ohne separaten Nenner muss die Eingabe als Funktionsausdruck erfolgen.")
        expr = _parse_symbolic_expression(num, substitutions=substitutions)
        numerator_expr, denominator_expr = sp.fraction(sp.together(expr))
        num_arr = _coerce_polynomial_coeffs(str(sp.expand(numerator_expr)), name='Zaehler')
        den_arr = _coerce_polynomial_coeffs(str(sp.expand(denominator_expr)), name='Nenner')
        return num_arr, den_arr

    return (
        _coerce_polynomial_coeffs(num, substitutions=substitutions, name='Zaehler'),
        _coerce_polynomial_coeffs(den, substitutions=substitutions, name='Nenner'),
    )


def _count_rhp_poles(den: np.ndarray) -> tuple[int, int]:
    poles = np.roots(den)
    rhp = int(np.sum(np.real(poles) > EPSILON))
    imag_axis = int(np.sum(np.isclose(np.real(poles), 0.0, atol=1e-8)))
    return rhp, imag_axis


def _nyquist_full_curve(num: np.ndarray,
                        den: np.ndarray,
                        w_min: float,
                        w_max: float,
                        punkte: int) -> tuple[np.ndarray, np.ndarray]:
    w_pos = np.logspace(np.log10(w_min), np.log10(w_max), punkte)
    sys = signal.TransferFunction(num.tolist(), den.tolist())
    _, resp_pos = signal.freqresp(sys, w=w_pos)
    # Allgemeiner Nyquist: kompletter Verlauf von -j*infty nach +j*infty.
    w_full = np.concatenate((-w_pos[::-1], w_pos))
    resp_full = np.concatenate((np.conj(resp_pos[::-1]), resp_pos))
    return w_full, resp_full


def _encirclements_about_minus_one(curve: np.ndarray) -> tuple[int, int]:
    shifted = curve + 1.0
    angles = np.unwrap(np.angle(shifted))
    delta = float(angles[-1] - angles[0])
    n_ccw = int(np.rint(delta / (2.0 * np.pi)))
    n_cw = -n_ccw
    return n_ccw, n_cw


def nyquist_kriterium(num: Sequence[float] | np.ndarray | str,
                      den: Sequence[float] | np.ndarray | str | None = None,
                      w_min: float = 1e-3,
                      w_max: float = 1e3,
                      punkte: int = 3000,
                      substitutions: Mapping[str, float] | None = None) -> Dict[str, Any]:
    """Bewertet Stabilität mit dem allgemeinen Nyquist-Kriterium.

    Verwendete Konvention:
    - N_cw: Uhrzeigersinn-Umschlingungen des Punkts -1+0j
    - P: Anzahl offener RHP-Pole von L(s)
    - Z: Anzahl geschlossener RHP-Pole aus 1+L(s)=0
    - Beziehung: Z = P + N_cw
    """
    num_arr, den_arr = parse_uebertragungsfunktion(num, den, substitutions=substitutions)

    if num_arr.size == 0 or den_arr.size < 2:
        raise ValueError("Zaehler und Nenner muessen gueltige Polynome sein.")
    if w_min <= 0 or w_max <= 0 or w_min >= w_max:
        raise ValueError("Es muss gelten: 0 < w_min < w_max.")
    if punkte < 300:
        raise ValueError("punkte muss mindestens 300 sein.")

    p_rhp, p_imag_axis = _count_rhp_poles(den_arr)
    w_full, curve = _nyquist_full_curve(num_arr, den_arr, w_min, w_max, punkte)
    n_ccw, n_cw = _encirclements_about_minus_one(curve)
    z_rhp = int(p_rhp + n_cw)
    stabil = bool(z_rhp == 0 and p_imag_axis == 0)

    min_dist_minus_one = float(np.min(np.abs(curve + 1.0)))
    hints: list[str] = []
    if p_imag_axis > 0:
        hints.append("Achtung: Offene Pole auf der imaginaeren Achse erkannt. Exakter Nyquist-Einzug ist separat zu behandeln.")
    if min_dist_minus_one < 1e-3:
        hints.append("Nyquist-Kurve verlaeuft sehr nah an -1. Numerische Aufloesung ggf. erhoehen.")

    weg = [
        {"title": "Offene Kreisuebertragungsfunktion", "math": "L(s) = N(s)/D(s)",
         "comment": "Die Stabilitaet des geschlossenen Kreises folgt aus dem Nyquist-Ortsverlauf von L(jw)."},
        {"title": "Offene RHP-Pole", "math": f"P = {p_rhp}",
         "comment": "P ist die Anzahl der Pole von L(s) in der rechten Halbebene."},
        {"title": "Umschlingungen von -1", "math": f"N_ccw = {n_ccw}, N_cw = {n_cw}",
         "comment": "N_cw zaehlt Uhrzeigersinn-Umschlingungen des kritischen Punkts -1."},
        {"title": "Nyquist-Beziehung", "math": f"Z = P + N_cw = {p_rhp} + {n_cw} = {z_rhp}",
         "comment": "Z entspricht den geschlossenen RHP-Polen. Stabil bei Z=0 (ohne Achspole)."},
    ]

    ergebnis = {
        "stabil": stabil,
        "P": p_rhp,
        "N_ccw": n_ccw,
        "N_cw": n_cw,
        "Z": z_rhp,
        "offene_pole_auf_imaginaerachse": p_imag_axis,
        "min_abstand_zu_minus_eins": min_dist_minus_one,
        "w": w_full,
        "nyquist": curve,
    }
    return {"ergebnis": ergebnis, "loesungsweg": weg, "hinweise": hints, "plot_pfad": None}


def hurwitz_kriterium(den: list[float] | list[int]) -> Dict[str, Any]:
    """Prueft die Stabilitaet des charakteristischen Polynoms mit dem Hurwitz-Kriterium."""
    a = np.array(den, dtype=float)
    if a.size < 2:
        raise ValueError("Das Polynom muss mindestens ersten Grades sein.")
    if a[0] <= 0:
        raise ValueError("Der Leitkoeffizient muss positiv sein.")
    a /= a[0]
    n = a.size - 1
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        offset = 1 if i % 2 == 0 else 0
        for j in range(n):
            index = 2 * j + offset
            H[i, j] = float(a[index]) if index < a.size else 0.0
    hauptminoren = [float(np.linalg.det(H[: m + 1, : m + 1])) for m in range(n)]
    stabil = all(minor > 0 for minor in hauptminoren)
    weg = [
        {"title": "Polynom normieren", "math": f"a = {a.tolist()}",
         "comment": "Der Leitkoeffizient wird auf 1 normiert, um die Hurwitz-Matrix zu bilden."},
        {"title": "Hurwitz-Matrix", "math": f"H =\n{H}",
         "comment": "Die Matrix wird aus den Koeffizienten des charakteristischen Polynoms gebildet."},
        {"title": "Hauptminoren berechnen", "math": f"det(H_1), ..., det(H_n) = {[round(minor, 6) for minor in hauptminoren]}",
         "comment": "Jede obere linke Untermatrix wird zur Stabilitätsbewertung herangezogen."},
        {"title": "Stabilitätsentscheidung", "comment": "Das System ist stabil, wenn alle Hauptminoren positiv sind.", "math": f"stabil = {stabil}"},
    ]
    return {"ergebnis": {"stabil": stabil, "hauptminoren": hauptminoren, "H": H}, "loesungsweg": weg, "plot_pfad": None}


def _ersatzzeile(vorige_zeile: np.ndarray, verbleibende_ordnung: int) -> np.ndarray:
    """Erzeugt die Ersatzzeile bei einer Nullzeile im Routh-Schema."""
    aktive_koeffizienten = vorige_zeile[vorige_zeile != 0]
    if aktive_koeffizienten.size == 0:
        return np.zeros_like(vorige_zeile)
    ableitung = np.array([float((aktive_koeffizienten.size - 1 - i) * aktive_koeffizienten[i])
                           for i in range(aktive_koeffizienten.size - 1)], dtype=float)
    result = np.zeros_like(vorige_zeile)
    result[: ableitung.size] = ableitung
    return result


def routh_kriterium(den: list[float] | list[int]) -> Dict[str, Any]:
    """Berechnet das Routh-Schema und zaehlt Vorzeichenwechsel in der ersten Spalte."""
    a = np.array(den, dtype=float)
    if a.size < 2:
        raise ValueError("Das Polynom muss mindestens ersten Grades sein.")
    if abs(a[0]) < EPSILON:
        raise ValueError("Der Leitkoeffizient darf nicht null sein.")
    zeilen = a.size
    spalten = (zeilen + 1) // 2
    tabelle = np.zeros((zeilen, spalten), dtype=float)
    tabelle[0, : len(a[0::2])] = a[0::2]
    tabelle[1, : len(a[1::2])] = a[1::2]
    for i in range(2, zeilen):
        if np.allclose(tabelle[i - 1, :], 0.0, atol=EPSILON):
            tabelle[i - 1, :] = _ersatzzeile(tabelle[i - 2, :], zeilen - i)
        for j in range(spalten - 1):
            a1 = tabelle[i - 2, 0]
            b1 = tabelle[i - 1, 0]
            if abs(b1) < EPSILON:
                b1 = EPSILON
            a2 = tabelle[i - 2, j + 1] if j + 1 < spalten else 0.0
            b2 = tabelle[i - 1, j + 1] if j + 1 < spalten else 0.0
            tabelle[i, j] = (b1 * a2 - a1 * b2) / b1
    erste_spalte = np.where(np.isclose(tabelle[:, 0], 0.0, atol=EPSILON), EPSILON, tabelle[:, 0])
    vorzeichenwechsel = int(np.sum(np.diff(np.sign(erste_spalte)) != 0))
    stabil = bool(vorzeichenwechsel == 0 and np.all(erste_spalte > 0))
    schema = pd.DataFrame(tabelle, columns=[f"c{j}" for j in range(spalten)])
    weg = [
        {"title": "Routh-Schema aufbauen", "math": f"Schema =\n{tabelle}",
         "comment": "Die Zeilen werden nach der Kreuzregel aus den Koeffizienten gebildet."},
        {"title": "Nullzeilen behandeln", "comment": "Nullzeilen werden durch die Ableitung der vorherigen Zeile ersetzt, um die Berechnung fortzusetzen."},
        {"title": "Erste Spalte", "math": f"{erste_spalte.tolist()}",
         "comment": "Die Vorzeichen in der ersten Spalte geben die Anzahl instabiler Pole an."},
        {"title": "Stabilitätsentscheidung", "math": f"Vorzeichenwechsel = {vorzeichenwechsel}",
         "comment": "Ein stabiles System hat keine Vorzeichenwechsel in der ersten Spalte."},
    ]
    return {"ergebnis": {"stabil": stabil, "schema": schema}, "loesungsweg": weg, "plot_pfad": None}
