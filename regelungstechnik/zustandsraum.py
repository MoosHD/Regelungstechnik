"""Zustandsraum- und Normalform-Werkzeuge."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from scipy import signal
from scipy.linalg import expm
import sympy as sp


def zustandsraum_zu_uebertragungsfunktion(A: list[list[float]] | np.ndarray,
                                           B: list[list[float]] | np.ndarray,
                                           C: list[list[float]] | np.ndarray,
                                           D: list[list[float]] | np.ndarray) -> Dict[str, Any]:
    A_arr, B_arr, C_arr, D_arr = map(lambda x: np.atleast_2d(np.array(x, dtype=float)), (A, B, C, D))
    num, den = signal.ss2tf(A_arr, B_arr, C_arr, D_arr)
    weg = [
        "Uebertragungsfunktion aus Zustandsraum: G(s) = C (sI-A)^{-1} B + D",
        f"A =\n{A_arr}",
        f"B =\n{B_arr}",
        f"C =\n{C_arr}",
        f"D =\n{D_arr}",
        f"Resultat: Zaehler = {num[0].tolist()}, Nenner = {den.tolist()}",
    ]
    return {"ergebnis": (num[0].tolist(), den.tolist()), "loesungsweg": weg, "plot_pfad": None}


def regelungsnormalform(num: list[float], den: list[float]) -> Dict[str, Any]:
    den_arr = np.array(den, dtype=float)
    num_arr = np.array(num, dtype=float)
    if den_arr[0] == 0:
        raise ValueError("Der hoechste Nennerkoeffizient darf nicht null sein.")
    n = den_arr.size - 1
    den_norm = den_arr / den_arr[0]
    a = den_norm[1:][::-1]
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
    A[-1, :] = -a
    B = np.zeros((n, 1), dtype=float)
    B[-1, 0] = 1.0
    num_pad = np.zeros(n, dtype=float)
    num_shift = np.pad(num_arr[::-1] / den_arr[0], (0, max(0, n - num_arr.size)))[:n]
    num_pad[: num_shift.size] = num_shift
    C = num_pad.reshape(1, -1)
    D = np.zeros((1, 1), dtype=float)
    weg = [
        "Erstellung der Regelungsnormalform aus dem Systempolynom.",
        f"A =\n{A}",
        f"B =\n{B}",
        f"C =\n{C}",
    ]
    return {"ergebnis": (A, B, C, D), "loesungsweg": weg, "plot_pfad": None}


def transitionsmatrix(A: list[list[float]] | np.ndarray,
                      t_werte: list[float]) -> Dict[str, Any]:
    A_arr = np.array(A, dtype=float)
    ergebnisse = [expm(A_arr * tv) for tv in t_werte]
    weg = [
        "Transitionsmatrix Phi(t)=exp(A*t) fuer die Systemmatrix A.",
        f"A =\n{A_arr}",
        f"Ausgewertet in Zeiten: {t_werte}",
    ]
    return {"ergebnis": ergebnisse, "loesungsweg": weg, "plot_pfad": None}


def transitionsmatrix_symbolisch(A: list[list[float]] | np.ndarray) -> Dict[str, Any]:
    A_sp = sp.Matrix(A)
    Phi = sp.simplify((A_sp * sp.symbols('t')).exp())
    weg = [
        "Symbolische Transitionsmatrix Phi(t) = exp(A*t).",
        f"A = {A_sp}",
        f"Phi(t) = {Phi}",
    ]
    return {"ergebnis": Phi, "loesungsweg": weg, "plot_pfad": None}


def jordan_normalform(A: list[list[float]] | np.ndarray) -> Dict[str, Any]:
    A_sp = sp.Matrix(A)
    P, J = A_sp.jordan_form()
    weg = [
        "Berechnung der Jordan-Normalform.",
        f"A = {A_sp}",
        f"P = {P}",
        f"J = {J}",
    ]
    return {"ergebnis": {"P": P, "J": J}, "loesungsweg": weg, "plot_pfad": None}


def poincare_klassifikation(A: list[list[float]] | np.ndarray) -> Dict[str, Any]:
    A_arr = np.array(A, dtype=float)
    det_A = float(np.linalg.det(A_arr))
    tr_A = float(np.trace(A_arr))
    diskriminante = tr_A ** 2 - 4 * det_A
    if det_A < 0:
        typ = "Sattelpunkt (instabil)"
    elif diskriminante > 0:
        typ = "Knoten"
    elif np.isclose(diskriminante, 0.0, atol=1e-12):
        typ = "Degenerierter Knoten"
    else:
        typ = "Spirale/Fokus"
    weg = [
        f"det(A) = {det_A:.6g}",
        f"tr(A) = {tr_A:.6g}",
        f"Diskriminante = {diskriminante:.6g}",
        f"Typ = {typ}",
    ]
    return {"ergebnis": {"typ": typ, "det": det_A, "trace": tr_A}, "loesungsweg": weg, "plot_pfad": None}
