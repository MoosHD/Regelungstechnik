"""Visualisierungsfunktionen fuer Regelungstechnik-Plots."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from regelungstechnik.stabilitaet import nyquist_kriterium

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def _neue_figur() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.grid(True, which="both", alpha=0.4)
    return fig, ax


def _speichern(fig: plt.Figure, name: str) -> str:
    pfad = PLOT_DIR / name
    fig.tight_layout()
    fig.savefig(pfad, dpi=150)
    plt.close(fig)
    return str(pfad)


def _clean_values(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float)
    result[np.isclose(result, 0.0, atol=1e-12)] = 0.0
    return result


def plot_sprungantwort(num: list[float], den: list[float], t_ende: float = 10.0,
                       dateiname: str = "sprungantwort.png") -> Dict[str, Any]:
    sys = signal.TransferFunction(num, den)
    t, y = signal.step(sys, T=np.linspace(0, t_ende, 1000))
    fig, ax = _neue_figur()
    ax.plot(t, y, label="Sprungantwort")
    ax.set_xlabel("Zeit t [s]")
    ax.set_ylabel("y(t)")
    ax.set_title("Sprungantwort")
    ax.legend(loc="best")
    ax.margins(0.02)
    pfad = _speichern(fig, dateiname)
    weg = [
        f"G(s) = {num}/{den}",
        "Sprungantwort des Systems auf einen Einheitssprung.",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": (t, y), "loesungsweg": weg, "plot_pfad": pfad}


def plot_impulsantwort(num: list[float], den: list[float], t_ende: float = 10.0,
                       dateiname: str = "impulsantwort.png") -> Dict[str, Any]:
    sys = signal.TransferFunction(num, den)
    t, y = signal.impulse(sys, T=np.linspace(0, t_ende, 1000))
    fig, ax = _neue_figur()
    ax.plot(t, y, label="Impulsantwort")
    ax.set_xlabel("Zeit t [s]")
    ax.set_ylabel("g(t)")
    ax.set_title("Impulsantwort")
    ax.legend(loc="best")
    ax.margins(0.02)
    pfad = _speichern(fig, dateiname)
    weg = [
        f"G(s) = {num}/{den}",
        "Impulsantwort als inverse Laplace-Transformation des Systems.",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": (t, y), "loesungsweg": weg, "plot_pfad": pfad}


def plot_bode(num: list[float], den: list[float], dateiname: str = "bode.png") -> Dict[str, Any]:
    sys = signal.TransferFunction(num, den)
    w, mag, phase = signal.bode(sys)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.semilogx(w, _clean_values(mag), label="Amplitude")
    ax2.semilogx(w, _clean_values(phase), label="Phase", color="tab:orange")
    ax1.set_ylabel("Amplitude [dB]")
    ax2.set_ylabel("Phase [deg]")
    ax2.set_xlabel("omega [rad/s]")
    ax1.set_title("Bode-Diagramm")
    ax1.grid(True, which="both", alpha=0.4)
    ax2.grid(True, which="both", alpha=0.4)
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    ax1.margins(x=0.02)
    ax2.margins(x=0.02)
    pfad = _speichern(fig, dateiname)
    weg = [
        f"G(s) = {num}/{den}",
        "Amplitudengang und Phasengang im Frequenzbereich.",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": (w, mag, phase), "loesungsweg": weg, "plot_pfad": pfad}


def plot_ortskurve(num: list[float], den: list[float], w_bereich: list[float] | None = None,
                    dateiname: str = "ortskurve.png") -> Dict[str, Any]:
    if w_bereich is None:
        w_bereich = np.logspace(-2, 3, 2000)
    sys = signal.TransferFunction(num, den)
    w, h = signal.freqresp(sys, w=w_bereich)
    fig, ax = _neue_figur()
    ax.plot(_clean_values(h.real), _clean_values(h.imag), label="Ortskurve")
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.plot(-1, 0, 'rx', markersize=8, label="kritischer Punkt")
    ax.set_xlabel("Re{G(jw)}")
    ax.set_ylabel("Im{G(jw)}")
    ax.set_title("Ortskurve")
    ax.legend(loc="best")
    ax.autoscale(enable=True)
    pfad = _speichern(fig, dateiname)
    weg = [
        f"G(s) = {num}/{den}",
        "Ortskurve im komplexen Koordinatensystem.",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": h, "loesungsweg": weg, "plot_pfad": pfad}


def plot_nyquist(num: list[float], den: list[float],
                 w_min: float = 1e-3,
                 w_max: float = 1e3,
                 punkte: int = 3000,
                 dateiname: str = "nyquist.png") -> Dict[str, Any]:
    analyse = nyquist_kriterium(num, den, w_min=w_min, w_max=w_max, punkte=punkte)
    ergebnis = analyse["ergebnis"]
    curve = np.array(ergebnis["nyquist"], dtype=complex)

    fig, ax = _neue_figur()
    ax.plot(_clean_values(curve.real), _clean_values(curve.imag), label="Nyquist")
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.plot(-1, 0, 'rx', markersize=8, label="kritischer Punkt -1")

    if curve.size >= 2:
        mid = curve.size // 2
        p0, p1 = curve[max(0, mid - 1)], curve[min(curve.size - 1, mid + 1)]
        ax.annotate("", xy=(p1.real, p1.imag), xytext=(p0.real, p0.imag),
                    arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.2))

    stabil_text = "stabil" if ergebnis["stabil"] else "instabil"
    ax.set_title(f"Nyquist-Ortskurve ({stabil_text})")
    ax.set_xlabel("Re{L(jw)}")
    ax.set_ylabel("Im{L(jw)}")
    ax.legend(loc="best")
    ax.autoscale(enable=True)
    # Gleiches Seitenverhaeltnis verhindert vertikales oder horizontales Stauchen.
    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(0.05)
    pfad = _speichern(fig, dateiname)

    weg = list(analyse.get("loesungsweg", []))
    weg.append(f"Plot gespeichert unter {pfad}")
    result = {
        "ergebnis": {
            "stabil": ergebnis["stabil"],
            "P": ergebnis["P"],
            "N_cw": ergebnis["N_cw"],
            "Z": ergebnis["Z"],
            "offene_pole_auf_imaginaerachse": ergebnis["offene_pole_auf_imaginaerachse"],
        },
        "loesungsweg": weg,
        "hinweise": analyse.get("hinweise", []),
        "plot_pfad": pfad,
    }
    return result


def plot_pol_nullstellen(num: list[float], den: list[float], dateiname: str = "pn_diagramm.png") -> Dict[str, Any]:
    nullstellen = np.roots(num)
    pole = np.roots(den)
    fig, ax = _neue_figur()
    ax.scatter(_clean_values(pole.real), _clean_values(pole.imag), marker='x', s=100, color='red', label='Pole')
    ax.scatter(_clean_values(nullstellen.real), _clean_values(nullstellen.imag), marker='o', s=100,
               facecolors='none', edgecolors='blue', label='Nullstellen')
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.set_xlabel("Re(s)")
    ax.set_ylabel("Im(s)")
    ax.set_title("Pol-Nullstellen-Diagramm")
    ax.legend(loc="best")
    ax.autoscale(enable=True)
    pfad = _speichern(fig, dateiname)
    stabil = bool(np.all(np.real(pole) < 0))
    weg = [
        f"Pole: {pole.tolist()}",
        f"Nullstellen: {nullstellen.tolist()}",
        f"Stabilitaet: {stabil}",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": {"pole": pole, "nullstellen": nullstellen, "stabil": stabil},
            "loesungsweg": weg, "plot_pfad": pfad}


def plot_wurzelortskurve(num: list[float], den: list[float], k_bereich: list[float] | None = None,
                         dateiname: str = "wurzelortskurve.png",
                         k_markierungen: Iterable[float] | None = None,
                         sigma_grenze: float | None = None,
                         daempfung_min: float | None = None,
                         omega_grenze: float | Iterable[float] | None = None,
                         asymptoten_anzeigen: bool = True) -> Dict[str, Any]:
    if k_bereich is None:
        k_bereich = np.linspace(0.01, 10.0, 500)
    num_arr = np.array(num, dtype=float)
    den_arr = np.array(den, dtype=float)
    m = max(len(den_arr), len(num_arr))
    num_pad = np.pad(num_arr, (m - len(num_arr), 0), mode='constant')
    den_pad = np.pad(den_arr, (m - len(den_arr), 0), mode='constant')
    alle_pole = np.empty((len(k_bereich), m - 1), dtype=complex)
    for idx, k in enumerate(k_bereich):
        den_k = np.polyadd(den_pad, k * num_pad)
        alle_pole[idx, :] = np.roots(den_k)
    fig, ax = _neue_figur()
    for pole_spur in alle_pole.T:
        ax.plot(_clean_values(pole_spur.real), _clean_values(pole_spur.imag), '-', linewidth=1)
    sc = ax.scatter(_clean_values(alle_pole.real).flatten(), _clean_values(alle_pole.imag).flatten(),
                    c=np.repeat(k_bereich, alle_pole.shape[1]), cmap='viridis', s=6)

    offene_pole = np.roots(den_arr)
    offene_nullstellen = np.roots(num_arr) if len(num_arr) > 1 else np.array([], dtype=complex)
    ax.scatter(offene_pole.real, offene_pole.imag, marker='x', s=80, color='red', label='Offene Pole')
    if len(offene_nullstellen) > 0:
        ax.scatter(offene_nullstellen.real, offene_nullstellen.imag, marker='o', s=80,
                   facecolors='none', edgecolors='blue', label='Offene Nullstellen')

    if asymptoten_anzeigen and len(offene_pole) > len(offene_nullstellen):
        q_count = len(offene_pole) - len(offene_nullstellen)
        sigma_a = ((np.sum(offene_pole) - np.sum(offene_nullstellen)) / q_count).real
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        r = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]), 1.0) * 1.2
        for q in range(q_count):
            angle = np.deg2rad((2 * q + 1) * 180.0 / q_count)
            x_line = np.array([sigma_a, sigma_a + r * np.cos(angle)])
            y_line = np.array([0.0, r * np.sin(angle)])
            ax.plot(x_line, y_line, '--', color='gray', linewidth=1)
        ax.scatter([sigma_a], [0.0], color='black', s=30, label='Asymptoten-Schwerpunkt')

    if sigma_grenze is not None:
        ax.axvline(float(sigma_grenze), color='tab:red', linestyle='--', linewidth=1.2,
                   label=f"Sigma-Grenze {sigma_grenze:.3g}")

    if daempfung_min is not None and 0.0 < daempfung_min < 1.0:
        theta = np.arccos(float(daempfung_min))
        r = max(np.max(np.abs(alle_pole.real)), np.max(np.abs(alle_pole.imag)), 1.0) * 1.2
        x = np.array([0.0, -r * np.cos(theta)])
        y = np.array([0.0, r * np.sin(theta)])
        ax.plot(x, y, ':', color='tab:orange', linewidth=1.2)
        ax.plot(x, -y, ':', color='tab:orange', linewidth=1.2,
                label=f"Dämpfungsgrenze zeta={daempfung_min:.3g}")

    if omega_grenze is not None:
        if isinstance(omega_grenze, (int, float, np.integer, np.floating)):
            omega_werte = [float(omega_grenze)]
        else:
            omega_werte = [float(w) for w in omega_grenze]
        omega_werte = sorted({abs(w) for w in omega_werte if abs(w) > 0.0})
        if omega_werte:
            omega_abs = max(omega_werte)
            ax.axhline(+omega_abs, color='tab:green', linestyle='--', linewidth=1.2,
                       label=f"Omega-Grenze |omega|={omega_abs:.3g}")
            ax.axhline(-omega_abs, color='tab:green', linestyle='--', linewidth=1.2)

    markierungen = list(k_markierungen) if k_markierungen is not None else []
    for k_mark in markierungen:
        idx = int(np.argmin(np.abs(np.array(k_bereich, dtype=float) - float(k_mark))))
        poles = alle_pole[idx, :]
        ax.scatter(poles.real, poles.imag, s=45, marker='D', label=f"k={float(k_bereich[idx]):.3g}")

    fig.colorbar(sc, ax=ax, label='k')
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.set_xlabel("Re(s)")
    ax.set_ylabel("Im(s)")
    ax.set_title("Wurzelortskurve")
    ax.legend(loc="best")
    ax.autoscale(enable=True)
    pfad = _speichern(fig, dateiname)
    weg = [
        f"Offene Kreisuebertragungsfunktion mit k als Parameter.",
        "Wurzelortskurve als Verlauf der Pole des geschlossenen Kreises.",
        f"Plot gespeichert unter {pfad}",
    ]
    return {"ergebnis": alle_pole, "loesungsweg": weg, "plot_pfad": pfad}


def plot_poincare(A: list[list[float]] | np.ndarray, dateiname: str = "poincare.png") -> str:
    A_arr = np.array(A, dtype=float)
    det_A = float(np.linalg.det(A_arr))
    tr_A = float(np.trace(A_arr))
    fig, ax = _neue_figur()
    tr_range = np.linspace(-10, 10, 400)
    ax.plot(tr_range, tr_range ** 2 / 4, 'k--', label="det = tr^2/4")
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='gray', lw=1)
    ax.scatter([tr_A], [det_A], color='red', label=f"System (tr={tr_A:.2f}, det={det_A:.2f})")
    ax.set_xlabel("tr(A)")
    ax.set_ylabel("det(A)")
    ax.set_title("Poincaré-Diagramm")
    ax.legend(loc='best')
    pfad = _speichern(fig, dateiname)
    return pfad
