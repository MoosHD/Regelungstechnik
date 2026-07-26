"""Berechnungen fuer Uebertragungsfunktionen und Regelkreisverhalten."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import sympy as sp
from scipy import signal


def _require_positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} muss > 0 sein.")
    return float(value)


def _build_pid_result(reglertyp: str, kp: float, ti: float | None, td: float | None,
                      verfahren: str, gueltig: bool) -> Dict[str, Any]:
    ki = kp / ti if ti and ti > 0 else 0.0
    kd = kp * td if td and td > 0 else 0.0
    return {
        "reglertyp": reglertyp,
        "verfahren": verfahren,
        "gueltig": gueltig,
        "Kp": float(kp),
        "Ki": float(ki),
        "Kd": float(kd),
        "Ti": float(ti) if ti is not None else None,
        "Td": float(td) if td is not None else None,
    }


def reglerparameter_nach_verfahren(reglertyp: str,
                                   verfahren: str,
                                   modus: str = "offen",
                                   K: float | None = None,
                                   T: float | None = None,
                                   K_T: float | None = None,
                                   K_krit: float | None = None,
                                   T_krit: float | None = None) -> Dict[str, Any]:
    """Berechnet P/PI/PID-Parameter nach Ziegler-Nichols oder Cohen-Coon.

    Parameter für offene Streckenmodelle:
    - K: statische Verstärkung der Strecke
    - T: Ausgleichszeitkonstante
    - K_T: Totzeit

    Parameter für Ziegler-Nichols geschlossen:
    - K_krit: kritische Verstärkung
    - T_krit: kritische Schwingungsdauer
    """
    rt = reglertyp.strip().upper()
    vf = verfahren.strip().lower()
    md = modus.strip().lower()

    if rt not in {"P", "PI", "PID"}:
        raise ValueError("Reglertyp muss P, PI oder PID sein.")
    if vf not in {"ziegler-nichols", "cohen-coon"}:
        raise ValueError("Verfahren muss 'ziegler-nichols' oder 'cohen-coon' sein.")

    weg: list[dict[str, Any]] = []
    hinweise: list[str] = []

    if vf == "ziegler-nichols" and md == "geschlossen":
        kk = _require_positive("K_krit", float(K_krit if K_krit is not None else 0.0))
        tk = _require_positive("T_krit", float(T_krit if T_krit is not None else 0.0))
        if rt == "P":
            kp, ti, td = 0.5 * kk, None, None
        elif rt == "PI":
            kp, ti, td = 0.45 * kk, tk / 1.2, None
        else:
            kp, ti, td = 0.6 * kk, 0.5 * tk, 0.125 * tk
        ergebnis = _build_pid_result(rt, kp, ti, td, "Ziegler-Nichols (geschlossen)", True)
        weg.extend([
            {"title": "Eingabewerte", "math": f"K_krit={kk}, T_krit={tk}",
             "comment": "Kritische Werte aus der Grenzstabilität werden verwendet."},
            {"title": "Reglerformeln", "comment": "Die Standard-ZN-Tabelle für den geschlossenen Kreis wird angewendet."},
            {"title": "Ergebnis", "math": f"Kp={ergebnis['Kp']:.6g}, Ki={ergebnis['Ki']:.6g}, Kd={ergebnis['Kd']:.6g}",
             "comment": "Zusätzlich werden Ti und Td ausgegeben, falls vorhanden."},
        ])
        hinweise.append("WolframAlpha-Tipp: solve characteristic equation for k to verify K_krit.")
        return {"ergebnis": ergebnis, "loesungsweg": weg, "hinweise": hinweise, "plot_pfad": None}

    k = _require_positive("K", float(K if K is not None else 0.0))
    t = _require_positive("T", float(T if T is not None else 0.0))
    kt = _require_positive("K_T", float(K_T if K_T is not None else 0.0))

    if vf == "ziegler-nichols":
        if md != "offen":
            raise ValueError("Für Ziegler-Nichols bitte modus='offen' oder 'geschlossen' wählen.")
        gueltig = kt < 0.5 * t
        if rt == "P":
            kp, ti, td = t / (k * kt), None, None
        elif rt == "PI":
            kp, ti, td = 0.9 * t / (k * kt), 3.33 * kt, None
        else:
            kp, ti, td = 1.2 * t / (k * kt), 2.0 * kt, 0.5 * kt
        ergebnis = _build_pid_result(rt, kp, ti, td, "Ziegler-Nichols (offen)", gueltig)
        weg.extend([
            {"title": "Eingabewerte", "math": f"K={k}, T={t}, K_T={kt}",
             "comment": "Strecke wird als PT1 mit Totzeit angenähert."},
            {"title": "Gültigkeitsbereich", "math": f"K_T < 0.5*T => {kt:.6g} < {0.5*t:.6g}",
             "comment": "Nur in diesem Bereich gelten die Standard-ZN-Formeln zuverlässig."},
            {"title": "Ergebnis", "math": f"Kp={ergebnis['Kp']:.6g}, Ki={ergebnis['Ki']:.6g}, Kd={ergebnis['Kd']:.6g}",
             "comment": "Die Reglerparameter wurden nach ZN (offen) berechnet."},
        ])
        if not gueltig:
            hinweise.append("Achtung: K_T >= 0.5*T. Ergebnis ist nur als grobe Startschätzung zu verwenden.")
    else:
        gueltig = kt < 2.0 * t
        r = kt / t
        if rt == "P":
            kp = (1.0 / k) * (1.0 / r) * (1.0 + r / 3.0)
            ti, td = None, None
        elif rt == "PI":
            kp = (1.0 / k) * (1.0 / r) * (0.9 + r / 12.0)
            ti = kt * (30.0 + 3.0 * r) / (9.0 + 20.0 * r)
            td = None
        else:
            kp = (1.0 / k) * (1.0 / r) * (4.0 / 3.0 + r / 4.0)
            ti = kt * (32.0 + 6.0 * r) / (13.0 + 8.0 * r)
            td = kt * 4.0 / (11.0 + 2.0 * r)
        ergebnis = _build_pid_result(rt, kp, ti, td, "Cohen-Coon", gueltig)
        weg.extend([
            {"title": "Eingabewerte", "math": f"K={k}, T={t}, K_T={kt}",
             "comment": "Cohen-Coon basiert ebenfalls auf einem Totzeit-PT1-Modell."},
            {"title": "Gültigkeitsbereich", "math": f"K_T < 2*T => {kt:.6g} < {2*t:.6g}",
             "comment": "Innerhalb dieses Bereichs ist die Näherung für viele Strecken robust."},
            {"title": "Ergebnis", "math": f"Kp={ergebnis['Kp']:.6g}, Ki={ergebnis['Ki']:.6g}, Kd={ergebnis['Kd']:.6g}",
             "comment": "Die Reglerparameter wurden nach Cohen-Coon berechnet."},
        ])
        if not gueltig:
            hinweise.append("Achtung: K_T >= 2*T. Cohen-Coon liegt außerhalb des empfohlenen Bereichs.")

    hinweise.append("WolframAlpha-Tipp: root locus (num)/(den) with gain k und gewünschten Dämpfungsgrad prüfen.")
    return {"ergebnis": ergebnis, "loesungsweg": weg, "hinweise": hinweise, "plot_pfad": None}


def phasenkorrekturglied_auslegung(typ: str,
                                   phi_grad: float,
                                   omega_c: float,
                                   K: float = 1.0) -> Dict[str, Any]:
    """Legt ein phasenanhebendes oder phasenabsenkendes Korrekturglied aus.

    Typen:
    - "anhebend": Gk(s) = K * (1 + v*T*s) / (1 + T*s), v > 1
    - "absenkend": Gk(s) = K * (1 + T*s) / (1 + v*T*s), v > 1
    """
    typ_norm = typ.strip().lower()
    if typ_norm not in {"anhebend", "absenkend"}:
        raise ValueError("typ muss 'anhebend' oder 'absenkend' sein.")
    if omega_c <= 0:
        raise ValueError("omega_c muss > 0 sein.")
    if K <= 0:
        raise ValueError("K muss > 0 sein.")

    phi_abs = abs(float(phi_grad))
    if phi_abs <= 0.0 or phi_abs >= 89.0:
        raise ValueError("phi_grad muss zwischen 0 und 89 Grad liegen.")

    phi_rad = np.deg2rad(phi_abs)
    v = float((1.0 + np.sin(phi_rad)) / (1.0 - np.sin(phi_rad)))
    T = float(1.0 / (omega_c * np.sqrt(v)))

    if typ_norm == "anhebend":
        num = [float(K * v * T), float(K)]
        den = [float(T), 1.0]
        phi_eff = phi_abs
        formula = "Gk(s) = K*(1+v*T*s)/(1+T*s)"
    else:
        num = [float(K * T), float(K)]
        den = [float(v * T), 1.0]
        phi_eff = -phi_abs
        formula = "Gk(s) = K*(1+T*s)/(1+v*T*s)"

    omega_z = float(1.0 / (v * T)) if typ_norm == "anhebend" else float(1.0 / T)
    omega_p = float(1.0 / T) if typ_norm == "anhebend" else float(1.0 / (v * T))

    weg = [
        {"title": "Zielvorgabe", "math": f"typ={typ_norm}, phi={phi_eff:.3g} deg, omega_c={omega_c:.6g}",
         "comment": "Die gewuenschte Phasenkorrektur wird am Auslegungsdurchtritt angegeben."},
        {"title": "Parameterfaktor", "math": f"v = (1+sin(phi))/(1-sin(phi)) = {v:.6g}",
         "comment": "Aus der maximalen Phasenanhebung/-absenkung ergibt sich der Faktor v."},
        {"title": "Zeitkonstante", "math": f"T = 1/(omega_c*sqrt(v)) = {T:.6g}",
         "comment": "Nullstelle und Pol werden um omega_c angeordnet."},
        {"title": "Korrekturglied", "math": formula,
         "comment": f"Ergebnis: num={num}, den={den}"},
    ]

    ergebnis = {
        "typ": typ_norm,
        "K": float(K),
        "phi_grad": float(phi_eff),
        "omega_c": float(omega_c),
        "v": v,
        "T": T,
        "omega_nullstelle": omega_z,
        "omega_pol": omega_p,
        "num": num,
        "den": den,
    }
    return {"ergebnis": ergebnis, "loesungsweg": weg, "plot_pfad": None}


def wurzelortsauslegung(num: list[float], den: list[float],
                       k_start: float = 0.0,
                       k_ende: float = 20.0,
                       anzahl_k: int = 81,
                       daempfung_min: float | None = None,
                       sigma_grenze: float | None = None) -> Dict[str, Any]:
    """Berechnet eine K-Tabelle und sinnvolle K-Kandidaten für die Wurzelortsauslegung."""
    if anzahl_k < 3:
        raise ValueError("anzahl_k muss mindestens 3 sein.")
    if k_ende <= k_start:
        raise ValueError("k_ende muss größer als k_start sein.")

    num_arr = np.array(num, dtype=float)
    den_arr = np.array(den, dtype=float)
    m = max(len(den_arr), len(num_arr))
    num_pad = np.pad(num_arr, (m - len(num_arr), 0), mode='constant')
    den_pad = np.pad(den_arr, (m - len(den_arr), 0), mode='constant')
    k_values = np.linspace(float(k_start), float(k_ende), int(anzahl_k))

    all_poles = np.empty((len(k_values), m - 1), dtype=complex)
    max_real = np.empty(len(k_values), dtype=float)
    min_zeta = np.zeros(len(k_values), dtype=float)
    stable = np.zeros(len(k_values), dtype=bool)

    for idx, k_val in enumerate(k_values):
        den_k = np.polyadd(den_pad, k_val * num_pad)
        poles = np.roots(den_k)
        all_poles[idx, :] = poles
        real_parts = poles.real
        max_real[idx] = float(np.max(real_parts)) if len(real_parts) else 0.0
        stable[idx] = bool(np.all(real_parts < 0.0))
        zeta_candidates: list[float] = []
        for p in poles:
            omega_n = abs(p)
            if omega_n <= 1e-12:
                continue
            zeta_candidates.append(float(max(0.0, -p.real / omega_n)))
        min_zeta[idx] = float(min(zeta_candidates)) if zeta_candidates else 0.0

    mask = np.ones(len(k_values), dtype=bool)
    if sigma_grenze is not None:
        mask &= max_real <= float(sigma_grenze)
    else:
        mask &= stable
    if daempfung_min is not None:
        mask &= min_zeta >= float(daempfung_min)

    filtered_idx = np.where(mask)[0]
    if len(filtered_idx) == 0:
        fallback = np.where(stable)[0]
        filtered_idx = fallback if len(fallback) else np.arange(len(k_values))

    sample_count = min(8, len(filtered_idx))
    if sample_count > 0:
        sample_positions = np.linspace(0, len(filtered_idx) - 1, sample_count).astype(int)
        recommended_idx = filtered_idx[sample_positions]
    else:
        recommended_idx = np.array([0])
    k_empfohlen = [float(k_values[i]) for i in recommended_idx]

    pole_tabelle: list[dict[str, Any]] = []
    for i in recommended_idx:
        pole_tabelle.append({
            "k": float(k_values[i]),
            "pole": [complex(p) for p in all_poles[i, :]],
            "max_real": float(max_real[i]),
            "min_daempfung": float(min_zeta[i]),
            "stabil": bool(stable[i]),
        })

    open_poles = np.roots(den_arr)
    open_zeros = np.roots(num_arr) if len(num_arr) > 1 else np.array([], dtype=complex)
    n = len(open_poles)
    z = len(open_zeros)
    asymptoten: list[dict[str, float]] = []
    centroid = None
    if n > z:
        centroid_val = (np.sum(open_poles) - np.sum(open_zeros)) / (n - z)
        centroid = float(np.real_if_close(centroid_val).real)
        for q in range(n - z):
            angle_deg = (2 * q + 1) * 180.0 / (n - z)
            asymptoten.append({"winkel_deg": float(angle_deg)})

    k_grenze = None
    for i in range(len(k_values) - 1):
        if max_real[i] <= 0.0 < max_real[i + 1]:
            a = max_real[i]
            b = max_real[i + 1]
            alpha = -a / (b - a) if abs(b - a) > 1e-12 else 0.0
            k_grenze = float(k_values[i] + alpha * (k_values[i + 1] - k_values[i]))
            break

    weg = [
        {"title": "Charakteristisches Polynom", "math": "den(s) + k*num(s) = 0",
         "comment": "Für jeden k-Wert werden die geschlossenen Pole numerisch bestimmt."},
        {"title": "Kandidatenwahl", "comment": "Empfohlene k-Werte erfüllen Stabilitäts- und ggf. Dämpfungs-/Sigma-Grenzen."},
        {"title": "Asymptoten", "math": f"Anzahl={max(n-z, 0)}, Schwerpunkt={centroid}",
         "comment": "Winkel und Schwerpunkt unterstützen die grafische Auslegung."},
    ]
    hints = [
        "WolframAlpha-Tipp: solve den(s)+k*num(s)=0 for s",
        "WolframAlpha-Tipp: plot Re[s(k)] und Im[s(k)] für die empfohlenen k-Werte.",
    ]

    ergebnis = {
        "k_empfohlen": k_empfohlen,
        "pole_tabelle": pole_tabelle,
        "k_grenze_stabilitaet": k_grenze,
        "asymptoten": asymptoten,
        "asymptoten_schwerpunkt": centroid,
        "offene_pole": [complex(p) for p in open_poles],
        "offene_nullstellen": [complex(z0) for z0 in open_zeros],
        "k_werte": [float(v) for v in k_values],
        "max_real": [float(v) for v in max_real],
        "min_daempfung": [float(v) for v in min_zeta],
    }
    return {"ergebnis": ergebnis, "loesungsweg": weg, "hinweise": hints, "plot_pfad": None}


def reihenschaltung(G1: tuple[list[float], list[float]],
                   G2: tuple[list[float], list[float]]) -> Dict[str, Any]:
    num1, den1 = G1
    num2, den2 = G2
    num = np.polymul(num1, num2).tolist()
    den = np.polymul(den1, den2).tolist()
    weg = [
        {"title": "Modellieren der Teilsysteme", "math": f"G1(s) = {num1}/{den1}, G2(s) = {num2}/{den2}",
         "comment": "Beide Übertragungsfunktionen werden für die Reihenschaltung vorbereitet."},
        {"title": "Multiplikation", "math": "G(s) = G1(s) * G2(s)",
         "comment": "In Reihe geschaltete Systeme multiplizieren ihre Zähler und Nenner."},
        {"title": "Resultat", "math": f"G(s) = {num}/{den}",
         "comment": "Die Gesamtübertragungsfunktion wurde berechnet."},
    ]
    return {"ergebnis": (num, den), "loesungsweg": weg, "plot_pfad": None}


def parallelschaltung(G1: tuple[list[float], list[float]],
                      G2: tuple[list[float], list[float]]) -> Dict[str, Any]:
    num1, den1 = G1
    num2, den2 = G2
    num = np.polyadd(np.polymul(num1, den2), np.polymul(num2, den1)).tolist()
    den = np.polymul(den1, den2).tolist()
    weg = [
        {"title": "Parallele Systeme", "math": f"G1(s) = {num1}/{den1}, G2(s) = {num2}/{den2}",
         "comment": "Bei Parallelschaltung werden die beiden Systeme addiert."},
        {"title": "Gesamtsystem", "math": "G(s) = G1(s) + G2(s)",
         "comment": "Der gemeinsame Nenner entsteht durch Multiplikation der Nenner."},
        {"title": "Resultat", "math": f"G(s) = {num}/{den}",
         "comment": "Die zusammengefasste Übertragungsfunktion ist berechnet."},
    ]
    return {"ergebnis": (num, den), "loesungsweg": weg, "plot_pfad": None}


def rueckkopplung(G_vorwaerts: tuple[list[float], list[float]],
                  G_rueckwaerts: tuple[list[float], list[float]],
                  negativ: bool = True) -> Dict[str, Any]:
    numv, denv = G_vorwaerts
    numr, denr = G_rueckwaerts
    num_offen = np.polymul(numv, numr)
    den_offen = np.polymul(denv, denr)
    vorzeichen = 1 if negativ else -1
    num_geschlossen = np.polymul(numv, denr).tolist()
    den_geschlossen = np.polyadd(np.polymul(denv, denr), vorzeichen * num_offen).tolist()
    art = "negative" if negativ else "positive"
    weg = [
        {"title": "Offener Kreislauf", "math": f"G0(s) = Gv(s) * Gr(s) = {num_offen}/{den_offen}",
         "comment": "Der offene Regelkreis wird als Produkt der Zweige modelliert."},
        {"title": "Geschlossener Kreis", "math": f"G(s) = {num_geschlossen}/{den_geschlossen}",
         "comment": f"Bei {art}er Rückkopplung wird der Nenner um G0(s) ergänzt."},
        {"title": "Ergebnis", "comment": "Die geschlossene Regelkreisübertragung ist bestimmt."},
    ]
    return {"ergebnis": (num_geschlossen, den_geschlossen), "loesungsweg": weg, "plot_pfad": None}


def _is_symbolic(value: Any) -> bool:
    if isinstance(value, str):
        return any(ch.isalpha() for ch in value)
    if isinstance(value, sp.Basic):
        return True
    if isinstance(value, (list, tuple)):
        return any(_is_symbolic(v) for v in value)
    return False


def _to_sympy_poly(coeffs: list[Any], s: sp.Symbol) -> sp.Poly:
    coerced = [sp.sympify(c) for c in coeffs]
    return sp.Poly(coerced, s)


def sprungantwort_mit_fex(G: tuple[list[float], list[float]],
                          F_ex: tuple[list[float], list[float]] | None = None,
                          t_ende: float = 10.0,
                          anzahl_punkte: int = 1000) -> Dict[str, Any]:
    """Berechnet die Sprungantwort der Gesamtübertragung G(s) * F_ex(s)."""
    num_g, den_g = G
    if F_ex is None:
        num_fex, den_fex = [1.0], [1.0]
    else:
        num_fex, den_fex = F_ex

    if _is_symbolic((num_g, den_g, num_fex, den_fex)):
        s = sp.symbols('s')
        num_poly_g = _to_sympy_poly(num_g, s)
        den_poly_g = _to_sympy_poly(den_g, s)
        num_poly_fex = _to_sympy_poly(num_fex, s)
        den_poly_fex = _to_sympy_poly(den_fex, s)
        num_total = sp.expand(num_poly_g.as_expr() * num_poly_fex.as_expr())
        den_total = sp.expand(den_poly_g.as_expr() * den_poly_fex.as_expr())
        s = sp.symbols('s')
        t = sp.symbols('t', positive=True)
        transfer = sp.simplify(num_total / den_total)
        step_response = sp.simplify(sp.inverse_laplace_transform(transfer / s, s, t))
        weg = [
            {"title": "Zusammengesetzte Übertragung", "math": f"G_gesamt(s) = G(s) * F_ex(s) = {sp.sstr(num_total)}/{sp.sstr(den_total)}",
             "comment": "Die Gesamtübertragung wurde symbolisch zusammengeführt."},
            {"title": "Einheitssprung", "comment": "Die Sprungantwort wird symbolisch für u(t)=1(t) beschrieben."},
            {"title": "Ergebnis", "math": f"y(t) = {sp.sstr(step_response)}", "comment": "Die Rücktransformation wurde symbolisch durchgeführt."},
        ]
        return {"ergebnis": {"status": "symbolisch", "num": sp.sstr(num_total), "den": sp.sstr(den_total), "y_t": sp.sstr(step_response)}, "loesungsweg": weg, "plot_pfad": None}

    num_total = np.polymul(num_g, num_fex).tolist()
    den_total = np.polymul(den_g, den_fex).tolist()
    t = np.linspace(0.0, t_ende, anzahl_punkte)
    sys = signal.TransferFunction(num_total, den_total)
    _, y = signal.step(sys, T=t)
    weg = [
        {"title": "Zusammengesetzte Übertragung", "math": f"G_gesamt(s) = G(s) * F_ex(s) = {num_total}/{den_total}",
         "comment": "Die Gesamtübertragung entsteht durch Multiplikation der Zähler- und Nennerpolynome."},
        {"title": "Einheitssprung", "comment": "Die Sprungantwort wird für u(t)=1(t) berechnet."},
        {"title": "Ergebnis", "math": "y(t) = L^{-1}{G_gesamt(s)/s}", "comment": "Die Antwort wird numerisch über die Transferfunktion berechnet."},
    ]
    return {"ergebnis": (t.tolist(), y.tolist()), "loesungsweg": weg, "plot_pfad": None}


def _step_response(num: list[float], den: list[float], t_ende: float = 20.0, anzahl_punkte: int = 5000):
    sys = signal.TransferFunction(num, den)
    t = np.linspace(0.0, t_ende, anzahl_punkte)
    _, y = signal.step(sys, T=t)
    return t, y


def stationaere_abweichung(num: list[float], den: list[float], eingangstyp: str = "sprung") -> Dict[str, Any]:
    """Berechnet die stationäre Abweichung für einen Sprunganforderung r(t)=1."""
    if eingangstyp != "sprung":
        raise ValueError("Nur 'sprung' als Eingabetyp wird derzeit unterstützt.")

    if _is_symbolic((num, den)):
        s = sp.symbols('s')
        G = _to_sympy_poly(num, s).as_expr() / _to_sympy_poly(den, s).as_expr()
        Y = G / s
        y_end = sp.limit(s * Y, s, 0)
        abweichung = sp.simplify(1 - y_end)
        weg = [
            {"title": "Übertragungsfunktion", "math": f"G(s) = {sp.sstr(G)}",
             "comment": "Die symbolische Übertragungsfunktion wird definiert."},
            {"title": "Einheitssprung", "math": "U(s) = 1/s",
             "comment": "Für eine Sprunganforderung wird das Eingangssignal symbolisch beschrieben."},
            {"title": "Endwertsatz", "math": "y(∞) = lim_{s->0} s*Y(s)",
             "comment": "Der stationäre Wert wird symbolisch mit dem Endwertsatz ermittelt."},
            {"title": "Ergebnis", "math": f"y(∞) = {sp.sstr(y_end)}, Abweichung = {sp.sstr(abweichung)}",
             "comment": "Die stationäre Abweichung zum Referenzwert 1 wurde berechnet."},
        ]
        return {"ergebnis": {"y_end": y_end, "abweichung": abweichung}, "loesungsweg": weg, "plot_pfad": None}

    if den[-1] == 0:
        raise ValueError("Die Übertragungsfunktion hat keinen endlichen stationären Endwert.")
    y_end = float(num[-1]) / float(den[-1])
    abweichung = 1.0 - y_end
    weg = [
        {"title": "DC-Gewinn", "math": f"G(0) = {num[-1]}/{den[-1]} = {y_end:.6f}",
         "comment": "Der Endwert für einen Sprung wird aus dem DC-Gewinn der Übertragungsfunktion gewonnen."},
        {"title": "Stationäre Abweichung", "math": f"e(∞) = 1 - y(∞) = {abweichung:.6f}",
         "comment": "Die stationäre Abweichung zum Sollwert 1 wird berechnet."},
    ]
    return {"ergebnis": {"y_end": y_end, "abweichung": abweichung}, "loesungsweg": weg, "plot_pfad": None}


def maximale_ueberschwingweite(num: list[float], den: list[float], t_ende: float = 20.0, anzahl_punkte: int = 5000) -> Dict[str, Any]:
    """Berechnet die maximale Überschwingweite der Sprungantwort relativ zum stationären Endwert."""
    t, y = _step_response(num, den, t_ende=t_ende, anzahl_punkte=anzahl_punkte)
    y_end = y[-1]
    if y_end == 0.0:
        ueberschwingweite = float(np.max(y))
    else:
        ueberschwingweite = (np.max(y) - y_end) / abs(y_end)
    weg = [
        {"title": "Sprungantwort berechnen", "math": f"G(s) = {num}/{den}",
         "comment": "Numerische Sprungantwort des Systems wird ermittelt."},
        {"title": "Stationärer Endwert", "math": f"y(∞) ≈ {y_end:.6f}",
         "comment": "Der letzte Wert des numerisch berechneten Signals dient als Näherung des Endwerts."},
        {"title": "Überschwingweite", "math": f"σ = (y_max - y(∞))/|y(∞)| = {ueberschwingweite:.6f}",
         "comment": "Die maximale Überschwingweite wird relativ zum stationären Wert angegeben."},
    ]
    return {"ergebnis": {"ueberschwingweite": float(ueberschwingweite), "y_end": float(y_end)},
            "loesungsweg": weg, "plot_pfad": None}


def ausregelzeit(num: list[float], den: list[float], toleranzband: float = 0.05,
                 t_ende: float = 20.0, anzahl_punkte: int = 5000) -> Dict[str, Any]:
    """Berechnet die Ausregelzeit der Sprungantwort für ein angegebenes Toleranzband."""
    t, y = _step_response(num, den, t_ende=t_ende, anzahl_punkte=anzahl_punkte)
    y_end = y[-1]
    band = toleranzband * abs(y_end)
    innerhalb = np.abs(y - y_end) <= band
    ausregelzeit_wert = 0.0
    for i in range(len(innerhalb) - 1, -1, -1):
        if not innerhalb[i]:
            ausregelzeit_wert = t[i + 1] if i + 1 < len(t) else t[-1]
            break
    weg = [
        {"title": "Sprungantwort berechnen", "math": f"G(s) = {num}/{den}",
         "comment": "Numerische Sprungantwort des Systems wird ermittelt."},
        {"title": "Toleranzband", "math": f"±{toleranzband*100:.1f}% von y(∞) = ±{band:.6f}",
         "comment": "Das zulässige Band um den stationären Wert wird festgelegt."},
        {"title": "Ausregelzeit", "math": f"t_s ≈ {ausregelzeit_wert:.6f} s",
         "comment": "Die letzte Zeit vor dem dauerhaften Verlassen des Toleranzbandes wird bestimmt."},
    ]
    return {"ergebnis": {"ausregelzeit": float(ausregelzeit_wert), "y_end": float(y_end)},
            "loesungsweg": weg, "plot_pfad": None}


def sprungfaehigkeit_realisierbarkeit(num: list[float], den: list[float]) -> Dict[str, Any]:
    grad_num = len(num) - 1
    grad_den = len(den) - 1
    sprungfaehig = grad_num == grad_den
    realisierbar = grad_num <= grad_den
    weg = [
        {"title": "Polynomgrade", "math": f"grad(num)={grad_num}, grad(den)={grad_den}",
         "comment": "Gradvergleich entscheidet über Sprungfähigkeit und Realisierbarkeit."},
        {"title": "Fallunterscheidung", "comment": "Sprungfähigkeit nur bei gleichem Grad, Kausalität bei geringerer oder gleichem Grad."},
        {"title": "Anwendung", "math": f"sprungfähig={sprungfaehig}, realisierbar={realisierbar}",
         "comment": "Die Systemeigenschaften des Reglers werden abschließend bewertet."},
    ]
    return {"ergebnis": {"sprungfaehig": sprungfaehig, "realisierbar": realisierbar},
            "loesungsweg": weg, "plot_pfad": None}
