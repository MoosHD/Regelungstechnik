"""Stabilitaetskriterien und Routh-Hurwitz-Berechnungen."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd

EPSILON = 1e-12


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
