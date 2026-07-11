"""
Regelungstechnik 1 - Toolkit
=============================
Automatisiert Berechnungen aus RT1 (TU Braunschweig, Pannek):
Laplace-Transformation, Partialbruchzerlegung, Zustandsraum,
Stabilitaetskriterien, Reglerentwurf und alle gängigen Plots.

Eingabeformat fuer Uebertragungsfunktionen:
    num, den = [1, 2], [1, 3, 2]   # G(s) = (s+2)/(s^2+3s+2), absteigende Potenzen

Jede Berechnungsfunktion liefert ein dict:
    {"ergebnis": ..., "loesungsweg": [str, ...], "plot_pfad": str oder None}
"""

import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.linalg import expm

s, t, k_sym = sp.symbols('s t k')

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _neue_figur():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.grid(True, which="both", alpha=0.4)
    return fig, ax


def _speichern(fig, name):
    pfad = os.path.join(PLOT_DIR, name)
    fig.tight_layout()
    fig.savefig(pfad, dpi=150)
    plt.close(fig)
    return pfad


def _print_ergebnis(titel, res):
    print(f"\n=== {titel} ===")
    print("Ergebnis:", res["ergebnis"])
    print("Lösungsweg:")
    for i, schritt in enumerate(res["loesungsweg"], 1):
        print(f"  {i}. {schritt}")
    if res.get("plot_pfad"):
        print("Plot gespeichert unter:", res["plot_pfad"])


# ---------------------------------------------------------------------------
# 1. Laplace-Transformation & Ruecktransformation
# ---------------------------------------------------------------------------

def laplace_transform(f_t):
    """Hin-Transformation f(t) -> F(s). f_t: sympy-Ausdruck in t."""
    F_s = sp.laplace_transform(f_t, t, s, noconds=True)
    weg = [
        f"Ausgangsfunktion im Zeitbereich: f(t) = {sp.sstr(f_t)}",
        "Anwendung der Laplace-Transformation L{f(t)} = Integral von 0 bis unendlich f(t) e^(-st) dt",
        f"Ergebnis im Bildbereich: F(s) = {sp.sstr(sp.simplify(F_s))}",
    ]
    return {"ergebnis": sp.simplify(F_s), "loesungsweg": weg, "plot_pfad": None}


def inverse_laplace(F_s):
    """Ruecktransformation F(s) -> f(t). F_s: sympy-Ausdruck in s."""
    f_t = sp.inverse_laplace_transform(F_s, s, t)
    weg = [
        f"Ausgangsfunktion im Bildbereich: F(s) = {sp.sstr(F_s)}",
        "Anwendung der inversen Laplace-Transformation via Bromwich-Integral / Korrespondenztabelle",
        f"Ergebnis im Zeitbereich: f(t) = {sp.sstr(sp.simplify(f_t))}",
    ]
    return {"ergebnis": sp.simplify(f_t), "loesungsweg": weg, "plot_pfad": None}


# ---------------------------------------------------------------------------
# 2. Partialbruchzerlegung
# ---------------------------------------------------------------------------

def partialbruchzerlegung(num, den):
    """Zerlegt G(s) = num(s)/den(s) in Partialbrueche."""
    Gs = sp.Poly(num, s).as_expr() / sp.Poly(den, s).as_expr()
    zerlegt = sp.apart(Gs, s)
    pole = sp.roots(sp.Poly(den, s))
    weg = [
        f"G(s) = ({sp.sstr(sp.Poly(num, s).as_expr())}) / ({sp.sstr(sp.Poly(den, s).as_expr())})",
        f"Nullstellen des Nenners (Pole): {pole}",
        "Ansatz: Summe von Termen c_i/(s-p_i) je Pol (bzw. hoehere Potenzen bei mehrfachen Polen)",
        "Koeffizienten c_i durch Koeffizientenvergleich bzw. Grenzwertbildung (Residuensatz) bestimmt",
        f"Ergebnis der Partialbruchzerlegung: {sp.sstr(zerlegt)}",
    ]
    return {"ergebnis": zerlegt, "loesungsweg": weg, "plot_pfad": None}


# ---------------------------------------------------------------------------
# 3. Uebertragungsfunktion & Blockschaltbildrechnung
# ---------------------------------------------------------------------------

def reihenschaltung(G1, G2):
    """G1, G2: (num, den) Tupel. Serie: G = G1 * G2."""
    num1, den1 = G1
    num2, den2 = G2
    num = np.polymul(num1, num2)
    den = np.polymul(den1, den2)
    weg = [
        f"G1(s) hat Zaehler {num1}, Nenner {den1}",
        f"G2(s) hat Zaehler {num2}, Nenner {den2}",
        "Reihenschaltung: G(s) = G1(s) * G2(s) -> Zaehler und Nenner werden multipliziert",
        f"Resultierender Zaehler: {list(num)}, Nenner: {list(den)}",
    ]
    return {"ergebnis": (list(num), list(den)), "loesungsweg": weg, "plot_pfad": None}


def parallelschaltung(G1, G2):
    """G1, G2: (num, den). Parallel: G = G1 + G2."""
    num1, den1 = G1
    num2, den2 = G2
    num = np.polyadd(np.polymul(num1, den2), np.polymul(num2, den1))
    den = np.polymul(den1, den2)
    weg = [
        f"G1(s) = {num1}/{den1}, G2(s) = {num2}/{den2}",
        "Parallelschaltung: G(s) = G1(s) + G2(s) -> gemeinsamer Nenner = den1 * den2",
        f"Zaehler = num1*den2 + num2*den1 = {list(num)}",
        f"Resultierender Nenner: {list(den)}",
    ]
    return {"ergebnis": (list(num), list(den)), "loesungsweg": weg, "plot_pfad": None}


def rueckkopplung(G_vorwaerts, G_rueckwaerts, negativ=True):
    """Geschlossener Kreis: G = Gv / (1 +/- Gv*Gr)."""
    numv, denv = G_vorwaerts
    numr, denr = G_rueckwaerts
    num_offen = np.polymul(numv, numr)
    den_offen = np.polymul(denv, denr)
    vorzeichen = 1 if negativ else -1
    num_geschlossen = np.polymul(numv, denr)
    den_geschlossen = np.polyadd(np.polymul(denv, denr), vorzeichen * num_offen)
    art = "negative" if negativ else "positive"
    weg = [
        f"Vorwaertszweig Gv(s) = {numv}/{denv}, Rueckwaertszweig Gr(s) = {numr}/{denr}",
        f"{art} Rueckkopplung: G(s) = Gv(s) / (1 {'+' if negativ else '-'} Gv(s)*Gr(s))",
        f"Kreisuebertragungsfunktion (offener Kreis) G0(s) = Gv*Gr: Zaehler {list(num_offen)}, Nenner {list(den_offen)}",
        f"Resultierender Zaehler geschlossener Kreis: {list(num_geschlossen)}",
        f"Resultierender Nenner geschlossener Kreis: {list(den_geschlossen)}",
    ]
    return {"ergebnis": (list(num_geschlossen), list(den_geschlossen)), "loesungsweg": weg, "plot_pfad": None}


def sprungfaehigkeit_realisierbarkeit(num, den):
    grad_zaehler = len(num) - 1
    grad_nenner = len(den) - 1
    sprungfaehig = grad_zaehler == grad_nenner
    realisierbar = grad_zaehler <= grad_nenner
    weg = [
        f"Grad des Zaehlerpolynoms: {grad_zaehler}",
        f"Grad des Nennerpolynoms: {grad_nenner}",
        "Sprungfaehig, wenn Grad(Zaehler) = Grad(Nenner) (Durchgriff D != 0)",
        "Realisierbar, wenn Grad(Zaehler) <= Grad(Nenner) (kausales System)",
        f"-> sprungfaehig: {sprungfaehig}, realisierbar: {realisierbar}",
    ]
    return {"ergebnis": {"sprungfaehig": sprungfaehig, "realisierbar": realisierbar},
            "loesungsweg": weg, "plot_pfad": None}


def zustandsraum_zu_uebertragungsfunktion(A, B, C, D):
    A, B, C, D = map(np.atleast_2d, (A, B, C, D))
    num, den = signal.ss2tf(A, B, C, D)
    weg = [
        "Uebertragungsfunktion aus Zustandsraum: G(s) = C (sI - A)^-1 B + D",
        f"Systemmatrix A =\n{A}",
        f"Eingangsmatrix B =\n{B}",
        f"Ausgangsmatrix C =\n{C}, Durchgriff D =\n{D}",
        f"Ergebnis: Zaehler = {num[0].tolist()}, Nenner = {den.tolist()}",
    ]
    return {"ergebnis": (num[0].tolist(), den.tolist()), "loesungsweg": weg, "plot_pfad": None}


def regelungsnormalform(num, den):
    """Erzeugt Zustandsraumdarstellung in Regelungsnormalform aus num/den."""
    den = np.array(den, dtype=float)
    num = np.array(num, dtype=float)
    n = len(den) - 1
    den_norm = den / den[0]
    a = den_norm[1:][::-1]  # a0..a_{n-1}
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
    A[-1, :] = -a
    B = np.zeros((n, 1))
    B[-1, 0] = 1
    num_pad = np.zeros(n)
    num_shift = num[::-1] / den[0]
    num_pad[:len(num_shift)] = num_shift
    C = num_pad.reshape(1, -1)
    D = np.array([[0.0]])
    weg = [
        f"Nennerkoeffizienten normiert (a_n=1): {den_norm.tolist()}",
        "Regelungsnormalform: A hat Einsen auf der oberen Nebendiagonale, letzte Zeile = -a_i (aus charakteristischem Polynom)",
        f"A =\n{A}",
        f"B = {B.flatten().tolist()} (Einheitsvektor am Ende)",
        f"C = {C.flatten().tolist()} (aus Zaehlerkoeffizienten)",
    ]
    return {"ergebnis": (A, B, C, D), "loesungsweg": weg, "plot_pfad": None}


# ---------------------------------------------------------------------------
# 4. Normalformen & Zustandsraum
# ---------------------------------------------------------------------------

def transitionsmatrix(A, t_werte):
    A = np.array(A, dtype=float)
    ergebnisse = [expm(A * tv) for tv in t_werte]
    weg = [
        f"Systemmatrix A =\n{A}",
        "Transitionsmatrix Phi(t) = exp(A*t), berechnet via Matrixexponential",
        f"Ausgewertet an den Zeitpunkten {t_werte}",
    ]
    return {"ergebnis": ergebnisse, "loesungsweg": weg, "plot_pfad": None}


def transitionsmatrix_symbolisch(A):
    A_sp = sp.Matrix(A)
    Phi = (A_sp * sp.Symbol('t')).exp()
    Phi = sp.simplify(Phi)
    weg = [
        f"Systemmatrix A = {A_sp.tolist()}",
        "Symbolische Berechnung von Phi(t) = exp(A*t) mittels Eigenwertzerlegung/Jordanform",
        f"Ergebnis: Phi(t) = {Phi}",
    ]
    return {"ergebnis": Phi, "loesungsweg": weg, "plot_pfad": None}


def jordan_normalform(A):
    A_sp = sp.Matrix(A)
    P, J = A_sp.jordan_form()
    weg = [
        f"Systemmatrix A = {A_sp.tolist()}",
        "Berechnung der Jordan-Normalform: A = P J P^-1",
        f"Transformationsmatrix P = {P.tolist()}",
        f"Jordan-Matrix J = {J.tolist()}",
    ]
    return {"ergebnis": {"P": P, "J": J}, "loesungsweg": weg, "plot_pfad": None}


def poincare_klassifikation(A):
    A = np.array(A, dtype=float)
    det_A = np.linalg.det(A)
    tr_A = np.trace(A)
    diskriminante = tr_A ** 2 - 4 * det_A
    if det_A < 0:
        typ = "Sattelpunkt (instabil)"
    elif diskriminante > 0:
        typ = "Knoten (Senke wenn tr<0, Quelle wenn tr>0)"
    elif diskriminante == 0:
        typ = "Degenerierter Knoten"
    else:
        if tr_A < 0:
            typ = "Stabile Spirale/Fokus"
        elif tr_A > 0:
            typ = "Instabile Spirale/Fokus"
        else:
            typ = "Zentrum (Kreisbahnen)"
    weg = [
        f"det(A) = {det_A:.4f}, tr(A) = {tr_A:.4f}",
        f"Diskriminante = tr(A)^2 - 4*det(A) = {diskriminante:.4f}",
        "Klassifikation nach Poincare-Diagramm (det(A), tr(A))-Ebene",
        f"-> Systemtyp: {typ}",
    ]
    return {"ergebnis": typ, "loesungsweg": weg, "plot_pfad": None}


def plot_poincare(A, dateiname="poincare.png"):
    A = np.array(A, dtype=float)
    det_A = np.linalg.det(A)
    tr_A = np.trace(A)
    fig, ax = _neue_figur()
    tr_range = np.linspace(-10, 10, 400)
    ax.plot(tr_range, tr_range ** 2 / 4, 'k--', label="det = tr^2/4 (Grenze Knoten/Spirale)")
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.scatter([tr_A], [det_A], color='red', zorder=5, label=f"System (tr={tr_A:.2f}, det={det_A:.2f})")
    ax.set_xlabel("tr(A)")
    ax.set_ylabel("det(A)")
    ax.set_title("Poincaré-Diagramm")
    ax.legend()
    pfad = _speichern(fig, dateiname)
    return pfad


# ---------------------------------------------------------------------------
# 5. Plots: Sprungantwort, Impulsantwort, Bode, Ortskurve, PN-Diagramm
# ---------------------------------------------------------------------------

def plot_sprungantwort(num, den, t_ende=10, dateiname="sprungantwort.png"):
    sys = signal.TransferFunction(num, den)
    t, y = signal.step(sys, T=np.linspace(0, t_ende, 1000))
    fig, ax = _neue_figur()
    ax.plot(t, y)
    ax.set_xlabel("Zeit t [s]")
    ax.set_ylabel("y(t)")
    ax.set_title("Sprungantwort")
    pfad = _speichern(fig, dateiname)
    weg = [
        f"Uebertragungsfunktion G(s) = {num}/{den}",
        "Sprungantwort = Antwort des Systems auf Heaviside-Sprung u(t)=1(t)",
        "Berechnung via scipy.signal.step (numerische Integration der Zustandsraumform)",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": (t, y), "loesungsweg": weg, "plot_pfad": pfad}


def plot_impulsantwort(num, den, t_ende=10, dateiname="impulsantwort.png"):
    sys = signal.TransferFunction(num, den)
    t, y = signal.impulse(sys, T=np.linspace(0, t_ende, 1000))
    fig, ax = _neue_figur()
    ax.plot(t, y)
    ax.set_xlabel("Zeit t [s]")
    ax.set_ylabel("g(t)")
    ax.set_title("Impulsantwort (Gewichtsfunktion)")
    pfad = _speichern(fig, dateiname)
    weg = [
        f"Uebertragungsfunktion G(s) = {num}/{den}",
        "Impulsantwort g(t) = inverse Laplace-Transformierte von G(s)",
        "Berechnung via scipy.signal.impulse",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": (t, y), "loesungsweg": weg, "plot_pfad": pfad}


def plot_bode(num, den, dateiname="bode.png"):
    sys = signal.TransferFunction(num, den)
    w, mag, phase = signal.bode(sys)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.semilogx(w, mag)
    ax1.set_ylabel("Amplitude [dB]")
    ax1.set_title("Bode-Diagramm")
    ax1.grid(True, which="both", alpha=0.4)
    ax2.semilogx(w, phase)
    ax2.set_ylabel("Phase [deg]")
    ax2.set_xlabel("Kreisfrequenz omega [rad/s]")
    ax2.grid(True, which="both", alpha=0.4)
    pfad = _speichern(fig, dateiname)
    knick = [1 / abs(r) for r in np.roots(den) if r != 0]
    weg = [
        f"G(s) = {num}/{den}",
        "Amplitudengang: |G(j*omega)| in dB = 20*log10(|G(j*omega)|)",
        "Phasengang: Winkel von G(j*omega) in Grad",
        f"Knickfrequenzen (1/|Zeitkonstante|) ungefaehr bei: {knick}",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": (w, mag, phase), "loesungsweg": weg, "plot_pfad": pfad}


def plot_ortskurve(num, den, w_bereich=None, dateiname="ortskurve.png"):
    if w_bereich is None:
        w_bereich = np.logspace(-2, 3, 2000)
    sys = signal.TransferFunction(num, den)
    w, h = signal.freqresp(sys, w=w_bereich)
    fig, ax = _neue_figur()
    ax.plot(h.real, h.imag)
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.plot(-1, 0, 'rx', markersize=10, label="kritischer Punkt (-1,0)")
    ax.set_xlabel("Re{G(j*omega)}")
    ax.set_ylabel("Im{G(j*omega)}")
    ax.set_title("Ortskurve (Polargang)")
    ax.legend()
    pfad = _speichern(fig, dateiname)
    weg = [
        f"G(s) = {num}/{den}",
        "Ortskurve = Graph von G(j*omega) in der komplexen Ebene fuer omega von 0 bis unendlich",
        "Kritischer Punkt (-1, 0) fuer Nyquist-Stabilitaetsbewertung eingezeichnet",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": h, "loesungsweg": weg, "plot_pfad": pfad}


def plot_pol_nullstellen(num, den, dateiname="pn_diagramm.png"):
    nullstellen = np.roots(num)
    pole = np.roots(den)
    fig, ax = _neue_figur()
    ax.scatter(pole.real, pole.imag, marker='x', s=100, color='red', label='Pole')
    ax.scatter(nullstellen.real, nullstellen.imag, marker='o', s=100,
               facecolors='none', edgecolors='blue', label='Nullstellen')
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.set_xlabel("Re(s)")
    ax.set_ylabel("Im(s)")
    ax.set_title("Pol-Nullstellen-Diagramm")
    ax.legend()
    pfad = _speichern(fig, dateiname)
    stabil = all(p.real < 0 for p in pole)
    weg = [
        f"Nullstellen (Wurzeln des Zaehlers): {nullstellen}",
        f"Pole (Wurzeln des Nenners): {pole}",
        "System EA-stabil, wenn alle Pole in der linken s-Halbebene (Re(s)<0) liegen",
        f"-> EA-stabil: {stabil}",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": {"pole": pole, "nullstellen": nullstellen, "stabil": stabil},
            "loesungsweg": weg, "plot_pfad": pfad}


def plot_wurzelortskurve(num, den, k_bereich=None, dateiname="wurzelortskurve.png"):
    if k_bereich is None:
        k_bereich = np.linspace(0.01, 10, 300)
    fig, ax = _neue_figur()
    alle_pole = []
    for k in k_bereich:
        den_k = np.polyadd(den, k * np.array(num + [0] * (len(den) - len(num))))
        pole = np.roots(den_k)
        alle_pole.append(pole)
    alle_pole = np.array(alle_pole)
    for i in range(alle_pole.shape[1]):
        sc = ax.scatter(alle_pole[:, i].real, alle_pole[:, i].imag, c=k_bereich, cmap='viridis', s=8)
    plt.colorbar(sc, ax=ax, label="Verstaerkung k")
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.set_xlabel("Re(s)")
    ax.set_ylabel("Im(s)")
    ax.set_title("Wurzelortskurve")
    pfad = _speichern(fig, dateiname)
    weg = [
        f"Offene Kreisuebertragungsfunktion: G0(s) = k * ({num}) / ({den})",
        "Wurzelortskurve = Verlauf der Pole des geschlossenen Kreises fuer k von 0 bis unendlich",
        "Geschlossener Nenner: den(s) + k*num(s), Nullstellen fuer jedes k berechnet",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": alle_pole, "loesungsweg": weg, "plot_pfad": pfad}


# ---------------------------------------------------------------------------
# 6. Stabilitaetskriterien
# ---------------------------------------------------------------------------

def hurwitz_kriterium(den):
    den = np.array(den, dtype=float)
    a = den / den[0]
    n = len(a) - 1
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            idx = 2 * (i + 1) - (j + 1) - 1
            if 0 <= idx <= n:
                coeff_idx = idx
                if 0 <= coeff_idx <= n:
                    H[i, j] = a[coeff_idx] if coeff_idx <= n else 0
    hauptminoren = [np.linalg.det(H[:m, :m]) for m in range(1, n + 1)]
    stabil = all(hm > 0 for hm in hauptminoren)
    weg = [
        f"Charakteristisches Polynom (normiert, a_n=1): {a.tolist()}",
        f"Hurwitz-Matrix (Groesse {n}x{n}) aus den Koeffizienten aufgebaut:\n{H}",
        f"Hauptminoren (Determinanten der oberen linken Teilmatrizen): {[round(h,4) for h in hauptminoren]}",
        "Hurwitz-Kriterium: System stabil, wenn ALLE Hauptminoren > 0",
        f"-> stabil: {stabil}",
    ]
    return {"ergebnis": {"stabil": stabil, "hauptminoren": hauptminoren, "H": H},
            "loesungsweg": weg, "plot_pfad": None}


def routh_kriterium(den):
    a = np.array(den, dtype=float)
    n = len(a) - 1
    m = n + 1
    ncols = (m + 1) // 2 + 1
    tabelle = np.zeros((m, ncols))
    tabelle[0, :len(a[0::2])] = a[0::2]
    tabelle[1, :len(a[1::2])] = a[1::2]
    epsilon = 1e-12
    for i in range(2, m):
        for j in range(ncols - 1):
            a1 = tabelle[i - 2, 0]
            b1 = tabelle[i - 1, 0]
            if abs(b1) < epsilon:
                b1 = epsilon
            a2 = tabelle[i - 2, j + 1] if j + 1 < ncols else 0
            b2 = tabelle[i - 1, j + 1] if j + 1 < ncols else 0
            tabelle[i, j] = (b1 * a2 - a1 * b2) / b1
    erste_spalte = tabelle[:, 0]
    vorzeichenwechsel = np.sum(np.diff(np.sign(erste_spalte)) != 0)
    stabil = vorzeichenwechsel == 0 and np.all(erste_spalte > 0)
    df = pd.DataFrame(tabelle, columns=[f"c{j}" for j in range(ncols)])
    weg = [
        f"Charakteristisches Polynom: {a.tolist()}",
        "Aufbau des Routh-Schemas: erste Zeile = Koeffizienten mit geradem Index, "
        "zweite Zeile = Koeffizienten mit ungeradem Index",
        "Folgezeilen ueber Kreuzregel: c_ij = (b1*a2 - a1*b2) / b1",
        f"Erste Spalte des Schemas: {erste_spalte.tolist()}",
        f"Anzahl Vorzeichenwechsel in erster Spalte: {vorzeichenwechsel} "
        "(entspricht Anzahl instabiler Pole in rechter Halbebene)",
        f"-> stabil: {stabil}",
    ]
    return {"ergebnis": {"stabil": stabil, "schema": df}, "loesungsweg": weg, "plot_pfad": None}


def nyquist_kriterium(num_offen, den_offen, offener_kreis_stabil=True, dateiname="nyquist.png"):
    pole_offen = np.roots(den_offen)
    P = np.sum(pole_offen.real > 0)
    w = np.concatenate([-np.logspace(3, -3, 2000), np.logspace(-3, 3, 2000)])
    sys = signal.TransferFunction(num_offen, den_offen)
    _, h = signal.freqresp(sys, w=w)
    winkel = np.unwrap(np.angle(h + 1))
    umschlingung = round((winkel[-1] - winkel[0]) / (2 * np.pi))
    N = -umschlingung
    Z = N + P
    stabil = Z == 0
    fig, ax = _neue_figur()
    ax.plot(h.real, h.imag)
    ax.plot(-1, 0, 'rx', markersize=10, label='-1')
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Nyquist-Ortskurve")
    ax.legend()
    pfad = _speichern(fig, dateiname)
    weg = [
        f"Offene Kreisuebertragungsfunktion G0(s) = {num_offen}/{den_offen}",
        f"Pole des offenen Kreises in rechter Halbebene: P = {P}",
        f"Anzahl Umschlingungen des Punktes -1 (im mathematisch positiven Sinn): N = {N}",
        "Nyquist-Kriterium: Z = N + P = Anzahl instabiler Pole des geschlossenen Kreises",
        f"-> Z = {Z}, geschlossener Kreis stabil: {stabil}",
    ]
    return {"ergebnis": {"stabil": stabil, "Z": Z, "N": N, "P": P}, "loesungsweg": weg, "plot_pfad": pfad}


def ea_stabilitaet(den):
    pole = np.roots(den)
    stabil = all(p.real < 0 for p in pole)
    weg = [
        f"Pole des Systems (Nullstellen des Nenners): {pole}",
        "EA-Stabilitaet: alle Pole muessen echten negativen Realteil haben",
        f"-> EA-stabil: {stabil}",
    ]
    return {"ergebnis": stabil, "loesungsweg": weg, "plot_pfad": None}


# ---------------------------------------------------------------------------
# 7. Reglerentwurf
# ---------------------------------------------------------------------------

def ziegler_nichols_offen(KP, T, KT):
    """Wendetangentenverfahren, offener Kreis. KP=Verstaerkung, T=Zeitkonstante, KT=Totzeit."""
    tabelle = {
        "P": {"KR": T / (KP * KT)},
        "PI": {"KR": 0.9 * T / (KP * KT), "TN": KT / 0.3},
        "PID": {"KR": 1.2 * T / (KP * KT), "TN": 2 * KT, "TV": 0.5 * KT},
    }
    weg = [
        f"Kennwerte aus Wendetangente: KP={KP}, T={T}, KT (Totzeit)={KT}",
        "Ziegler-Nichols offener Kreis nutzt PT1Tt-Approximation der Sprungantwort",
        f"P-Regler: KR = T/(KP*KT) = {tabelle['P']['KR']:.4f}",
        f"PI-Regler: KR = 0.9*T/(KP*KT) = {tabelle['PI']['KR']:.4f}, TN = KT/0.3 = {tabelle['PI']['TN']:.4f}",
        f"PID-Regler: KR = 1.2*T/(KP*KT) = {tabelle['PID']['KR']:.4f}, "
        f"TN = 2*KT = {tabelle['PID']['TN']:.4f}, TV = 0.5*KT = {tabelle['PID']['TV']:.4f}",
    ]
    return {"ergebnis": tabelle, "loesungsweg": weg, "plot_pfad": None}


def ziegler_nichols_geschlossen(K_kritisch, T_kritisch):
    tabelle = {
        "P": {"KR": 0.5 * K_kritisch},
        "PI": {"KR": 0.45 * K_kritisch, "TN": 0.85 * T_kritisch},
        "PID": {"KR": 0.6 * K_kritisch, "TN": 0.5 * T_kritisch, "TV": 0.125 * T_kritisch},
    }
    weg = [
        f"Kritische Verstaerkung K_krit = {K_kritisch}, kritische Periodendauer T_krit = {T_kritisch}",
        "Ziegler-Nichols geschlossener Kreis (Dauerschwingversuch)",
        f"P: KR = 0.5*K_krit = {tabelle['P']['KR']:.4f}",
        f"PI: KR = 0.45*K_krit = {tabelle['PI']['KR']:.4f}, TN = 0.85*T_krit = {tabelle['PI']['TN']:.4f}",
        f"PID: KR = 0.6*K_krit = {tabelle['PID']['KR']:.4f}, TN = 0.5*T_krit = {tabelle['PID']['TN']:.4f}, "
        f"TV = 0.125*T_krit = {tabelle['PID']['TV']:.4f}",
    ]
    return {"ergebnis": tabelle, "loesungsweg": weg, "plot_pfad": None}


def cohen_coon(KP, T, KT):
    tau_ratio = KT / T
    tabelle = {
        "P": {"KR": (T / (KP * KT)) * (1 + tau_ratio / 3)},
        "PI": {"KR": (T / (KP * KT)) * (0.9 + tau_ratio / 12),
               "TN": KT * (30 + 3 * tau_ratio) / (9 + 20 * tau_ratio)},
        "PID": {"KR": (T / (KP * KT)) * (4 / 3 + tau_ratio / 4),
                "TN": KT * (32 + 6 * tau_ratio) / (13 + 8 * tau_ratio),
                "TV": KT * 4 / (11 + 2 * tau_ratio)},
    }
    weg = [
        f"Kennwerte: KP={KP}, T={T}, KT={KT}, Verhaeltnis KT/T = {tau_ratio:.4f}",
        "Cohen-Coon-Methode (verbesserte Reaktionskurven-Methode fuer Totzeit-behaftete Systeme)",
        f"P-Regler: KR = {tabelle['P']['KR']:.4f}",
        f"PI-Regler: KR = {tabelle['PI']['KR']:.4f}, TN = {tabelle['PI']['TN']:.4f}",
        f"PID-Regler: KR = {tabelle['PID']['KR']:.4f}, TN = {tabelle['PID']['TN']:.4f}, TV = {tabelle['PID']['TV']:.4f}",
    ]
    return {"ergebnis": tabelle, "loesungsweg": weg, "plot_pfad": None}


def beurteilungskriterien_sprungantwort(num, den, t_ende=20, toleranzband=0.05):
    sys = signal.TransferFunction(num, den)
    t, y = signal.step(sys, T=np.linspace(0, t_ende, 5000))
    y_end = y[-1]
    ueberschwingweite = (max(y) - y_end) / y_end if y_end != 0 else max(y)
    innerhalb = np.abs(y - y_end) <= toleranzband * abs(y_end)
    ausregelzeit = None
    for i in range(len(innerhalb) - 1, -1, -1):
        if not innerhalb[i]:
            ausregelzeit = t[i + 1] if i + 1 < len(t) else t[-1]
            break
    if ausregelzeit is None:
        ausregelzeit = 0.0
    regelflaeche = np.trapz(np.abs(y_end - y), t)
    weg = [
        f"Sprungantwort numerisch berechnet fuer G(s) = {num}/{den}",
        f"Stationaerer Endwert y(inf) = {y_end:.4f}",
        f"Maximale Ueberschwingweite = (y_max - y_end)/y_end = {ueberschwingweite*100:.2f}%",
        f"Ausregelzeit (Verlassen des {int(toleranzband*100)}%-Toleranzbandes) = {ausregelzeit:.3f} s",
        f"Regelflaeche = Integral |y_end - y(t)| dt = {regelflaeche:.4f}",
    ]
    return {"ergebnis": {"y_end": y_end, "ueberschwingweite": ueberschwingweite,
                          "ausregelzeit": ausregelzeit, "regelflaeche": regelflaeche},
            "loesungsweg": weg, "plot_pfad": None}


def pt2_approximation(dominante_pole):
    """dominante_pole: Liste zweier komplex konjugierter Pole s = -sigma +/- j*omega_d."""
    p = dominante_pole[0]
    sigma = -p.real
    omega_d = abs(p.imag)
    omega_n = np.sqrt(sigma ** 2 + omega_d ** 2)
    D = sigma / omega_n
    ueberschwingweite = np.exp(-np.pi * D / np.sqrt(1 - D ** 2)) if D < 1 else 0
    weg = [
        f"Dominantes Polpaar: s = {-sigma:.4f} +/- j*{omega_d:.4f}",
        "Eigenfrequenz omega_n = sqrt(sigma^2 + omega_d^2)",
        f"omega_n = {omega_n:.4f} rad/s",
        "Daempfung D = sigma / omega_n",
        f"D = {D:.4f}",
        f"Ueberschwingweite (PT2-Naeherung) = exp(-pi*D/sqrt(1-D^2)) = {ueberschwingweite*100:.2f}%",
    ]
    return {"ergebnis": {"omega_n": omega_n, "D": D, "ueberschwingweite": ueberschwingweite},
            "loesungsweg": weg, "plot_pfad": None}


# ---------------------------------------------------------------------------
# 8. Fuehrungs-/Stoerverhalten & stationaere Genauigkeit
# ---------------------------------------------------------------------------

def fuehrungs_stoerverhalten(num_regler, den_regler, num_strecke, den_strecke):
    num_offen = np.polymul(num_regler, num_strecke)
    den_offen = np.polymul(den_regler, den_strecke)
    fuehrung = rueckkopplung((num_regler, den_regler), (num_strecke, den_strecke), negativ=True)
    num_stoer = np.polymul(num_strecke, den_regler)
    den_stoer = np.polyadd(np.polymul(den_regler, den_strecke), num_offen)
    weg = [
        "Fuehrungsuebertragungsfunktion Gw(s) = Gr*Gs / (1 + Gr*Gs)",
        "Stoeruebertragungsfunktion Gz(s) = Gs / (1 + Gr*Gs)",
        f"Fuehrung: Zaehler {fuehrung['ergebnis'][0]}, Nenner {fuehrung['ergebnis'][1]}",
        f"Stoerung: Zaehler {list(num_stoer)}, Nenner {list(den_stoer)}",
    ]
    return {"ergebnis": {"fuehrung": fuehrung["ergebnis"], "stoerung": (list(num_stoer), list(den_stoer))},
            "loesungsweg": weg, "plot_pfad": None}


def stationaere_genauigkeit(num, den, eingangstyp="sprung"):
    """Endwertsatz: lim s->0 s*Y(s). eingangstyp: 'sprung' (1/s) oder 'rampe' (1/s^2)."""
    s_sym = sp.Symbol('s')
    G = sp.Poly(num, s_sym).as_expr() / sp.Poly(den, s_sym).as_expr()
    eingang = 1 / s_sym if eingangstyp == "sprung" else 1 / s_sym ** 2
    Y = G * eingang
    endwert = sp.limit(s_sym * Y, s_sym, 0)
    weg = [
        f"G(s) = {sp.sstr(G)}, Eingangssignal: {eingangstyp} -> U(s) = {sp.sstr(eingang)}",
        "Endwertsatz: y(inf) = lim (s->0) s*Y(s) = lim (s->0) s*G(s)*U(s)",
        f"Ergebnis: y(inf) = {endwert}",
    ]
    return {"ergebnis": endwert, "loesungsweg": weg, "plot_pfad": None}


# ---------------------------------------------------------------------------
# Beispielaufruf (Feder-Masse-Daempfer-System)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    m, d, c = 1.0, 2.0, 5.0
    A = [[0, 1], [-c / m, -d / m]]
    B = [[0], [1 / m]]
    C = [[1, 0]]
    D = [[0]]

    print("\n########## BEISPIEL: Feder-Masse-Daempfer-System ##########")
    res = zustandsraum_zu_uebertragungsfunktion(A, B, C, D)
    _print_ergebnis("Uebertragungsfunktion aus Zustandsraum", res)
    num, den = res["ergebnis"]

    res = partialbruchzerlegung([round(x, 4) for x in num], [round(x, 4) for x in den])
    _print_ergebnis("Partialbruchzerlegung", res)

    res = plot_sprungantwort(num, den)
    _print_ergebnis("Sprungantwort", res)

    res = plot_bode(num, den)
    _print_ergebnis("Bode-Diagramm", res)

    res = plot_ortskurve(num, den)
    _print_ergebnis("Ortskurve", res)

    res = plot_pol_nullstellen(num, den)
    _print_ergebnis("Pol-Nullstellen-Diagramm", res)

    res = hurwitz_kriterium(den)
    _print_ergebnis("Hurwitz-Kriterium", res)

    res = routh_kriterium(den)
    _print_ergebnis("Routh-Kriterium", res)
    print(res["ergebnis"]["schema"])

    res = ea_stabilitaet(den)
    _print_ergebnis("EA-Stabilitaet", res)

    res = beurteilungskriterien_sprungantwort(num, den)
    _print_ergebnis("Beurteilungskriterien", res)

    res = poincare_klassifikation(A)
    _print_ergebnis("Poincare-Klassifikation", res)
    plot_poincare(A)

    res = ziegler_nichols_offen(KP=2.0, T=3.0, KT=0.5)
    _print_ergebnis("Ziegler-Nichols (offener Kreis)", res)

    res = plot_wurzelortskurve(num, den)
    _print_ergebnis("Wurzelortskurve", res)

    print("\nAlle Plots liegen im Ordner 'plots/'.")