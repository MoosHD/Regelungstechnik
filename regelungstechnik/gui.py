"""GUI fuer das Regelungstechnik-Toolkit."""
from __future__ import annotations
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, List
import numpy as np
import sympy as sp

from regelungstechnik.laplace import inverse_laplace, laplace_transform, partialbruchzerlegung
from regelungstechnik.plots import (plot_bode, plot_impulsantwort, plot_ortskurve,
                                    plot_pol_nullstellen, plot_wurzelortskurve,
                                    plot_sprungantwort)
from regelungstechnik.reglerentwurf import (parallelschaltung, reihenschaltung,
                                           rueckkopplung, sprungantwort_mit_fex,
                                           sprungfaehigkeit_realisierbarkeit,
                                           reglerparameter_nach_verfahren,
                                           wurzelortsauslegung)
from regelungstechnik.stabilitaet import hurwitz_kriterium, routh_kriterium

FunctionSpec = Dict[str, Any]


def _parse_polynom(text: str) -> List[float]:
    values = [x.strip() for x in text.replace(';', ',').split(',') if x.strip()]
    return [float(sp.nsimplify(sp.sympify(val))) for val in values]


def _parse_expression(text: str) -> sp.Expr:
    return sp.sympify(text, evaluate=True)


def _parse_optional_float(text: str) -> float | None:
    value = text.strip()
    if value == "":
        return None
    return float(value)


def _parse_transfer_function_expression(text: str) -> tuple[list[Any], list[Any]]:
    expr = sp.sympify(text, evaluate=True)
    s = sp.symbols('s')

    num, den = sp.fraction(sp.together(expr))
    num_poly = sp.Poly(sp.expand(num), s)
    den_poly = sp.Poly(sp.expand(den), s)
    coeffs_num = _normalize_coeffs([sp.simplify(c) for c in num_poly.all_coeffs()])
    coeffs_den = _normalize_coeffs([sp.simplify(c) for c in den_poly.all_coeffs()])
    return coeffs_num, coeffs_den


def _coerce_numeric_coeffs(coeffs: list[Any]) -> list[float]:
    numeric_coeffs: List[float] = []
    for coeff in coeffs:
        if isinstance(coeff, sp.Basic):
            value = sp.nsimplify(coeff)
            if value.free_symbols:
                raise ValueError(f"Symbolische Koeffizienten sind nicht numerisch: {coeff}")
            numeric_coeffs.append(float(value))
        else:
            numeric_coeffs.append(float(coeff))
    return numeric_coeffs


def _normalize_coeffs(coeffs: list[Any]) -> list[Any]:
    normalized: List[Any] = []
    for coeff in coeffs:
        if isinstance(coeff, sp.Basic):
            value = sp.nsimplify(coeff)
            if value.free_symbols:
                normalized.append(value)
            else:
                normalized.append(float(value))
        else:
            normalized.append(float(coeff))
    return normalized


def _format_scalar(value: Any) -> str:
    if isinstance(value, (bool, int, str)):
        return str(value)
    if isinstance(value, float):
        try:
            return str(sp.nsimplify(value))
        except Exception:
            return f"{value:.12g}"
    if isinstance(value, complex):
        real = sp.nsimplify(value.real)
        imag = sp.nsimplify(value.imag)
        return f"{real} + {imag}*I"
    try:
        expr = sp.nsimplify(value)
        return str(expr)
    except Exception:
        return str(value)


def _format_result(value: Any, depth: int = 0) -> str:
    if depth > 3:
        return "..."
    if value is None:
        return "None"
    if isinstance(value, (bool, int, str, float, complex)):
        return _format_scalar(value)
    if isinstance(value, sp.Expr):
        return str(sp.simplify(value))
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return f"ndarray(shape={value.shape})"
        if value.ndim == 1 and value.size <= 10:
            return "[" + ", ".join(_format_scalar(v) for v in value.tolist()) + "]"
        return f"ndarray(shape={value.shape})"
    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            sample = ", ".join(_format_result(v, depth + 1) for v in value[:10])
            return f"[{sample}, ...] ({len(value)} items)"
        return "[" + ", ".join(_format_result(v, depth + 1) for v in value) + "]"
    if isinstance(value, dict):
        items = [f"{k}: {_format_result(v, depth + 1)}" for k, v in value.items()]
        return "{ " + ", ".join(items) + " }"
    try:
        return str(value)
    except Exception:
        return repr(value)


def _format_step(step: Any) -> str:
    if isinstance(step, dict):
        lines: List[str] = []
        title = step.get("title")
        math = step.get("math")
        comment = step.get("comment")
        if title:
            lines.append(f"{title}")
        if math:
            lines.append(f"  {math}")
        if comment:
            lines.append(f"  {comment}")
        return "\n".join(lines)
    return str(step)


def _build_output(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Ergebnis:")
    lines.append(_format_result(result.get("ergebnis")))

    loesungsweg = result.get("loesungsweg") or []
    if loesungsweg:
        # Bei Fehlern explizit anzeigen statt nur "None" auszugeben.
        if all(isinstance(step, str) and step.lower().startswith("fehler") for step in loesungsweg):
            lines.append("\nFehler:")
            for step in loesungsweg:
                lines.append(f"- {step}")
        else:
            lines.append("\nLösungsweg:")
            for idx, step in enumerate(loesungsweg[:3], start=1):
                lines.append(f"{idx}. {_format_step(step)}")

    hinweise = result.get("hinweise")
    if hinweise:
        lines.append("\nHinweise:")
        for hint in hinweise:
            lines.append(f"- {hint}")
    plot_path = result.get("plot_pfad")
    if plot_path:
        lines.append("\nPlot gespeichert unter:")
        lines.append(str(plot_path))
    return "\n".join(lines)


def _guidance_for_result(name: str, result: Dict[str, Any]) -> str:
    suggestions: List[str] = []
    if name == "Laplace-Transformation":
        suggestions.append("Nutzen Sie die transformierte Funktion zur Analyse im s-Bereich.")
        suggestions.append("Als nächstes: Prüfen Sie Stabilität per Hurwitz oder Routh.")
    elif name == "Inverse Laplace":
        suggestions.append("Verwenden Sie die Zeitfunktion für Sprung- oder Impulsantworten.")
        suggestions.append("Zeigen Sie das Zeitverhalten mit einem Plot der Schrittantwort.")
    elif name == "Partialbruchzerlegung":
        suggestions.append("Nutzen Sie die Teilbruchzerlegung zur inversen Laplace-Transformation.")
        suggestions.append("Sehen Sie nach, ob die Terme handlich für Regelkreise sind.")
    elif name in {"Hurwitz-Kriterium", "Routh-Kriterium"}:
        stabil = result.get("ergebnis", {}).get("stabil")
        if stabil is True:
            suggestions.append("Das System ist stabil. Als nächsten Schritt prüfen Sie Frequenzgang oder Zeitantwort.")
        elif stabil is False:
            suggestions.append("Das System ist instabil. Entwerfen Sie einen Regelkreis mit Rückkopplung.")
        suggestions.append("Verwenden Sie Pole/Nullstellen-Plot oder Wurzelortskurve zur Visualisierung.")
    elif name in {"Reihenschaltung", "Parallelschaltung", "Rueckkopplung"}:
        suggestions.append("Bestimmen Sie anschließend die Stabilität des Gesamtsystems.")
        suggestions.append("Führen Sie eine Sprungantwort oder Bode-Analyse zur Validierung durch.")
    elif name == "Sprungfaehigkeit":
        suggestions.append("Wenn springfähig, können Sie den Regelkreis auf Referenzverfolgung prüfen.")
        suggestions.append("Achten Sie auf eine kauseal realisierbare Übertragungsfunktion.")
    elif name == "Reglerentwurf (ZN/CC)":
        suggestions.append("Parameter als Startwert nutzen und mit Zeit-/Frequenzantwort validieren.")
        suggestions.append("Für die Feinabstimmung Grenzstabilität und Wurzelortskurve zusätzlich prüfen.")
    elif name == "Wurzelortsauslegung":
        suggestions.append("Nutzen Sie die empfohlenen k-Werte als Kandidaten für den Reglerverstärkungsfaktor.")
        suggestions.append("Dämpfungs- und Sigma-Grenzen helfen bei der robusten Auswahl.")
    return "\n".join(suggestions)


def _workflow_explanation(name: str) -> str:
    explanations: Dict[str, str] = {
        "Laplace-Transformation": (
            "Was suche ich:\n"
            "- Die Bildfunktion F(s) zu einer Zeitfunktion f(t).\n\n"
            "Was brauche ich:\n"
            "- Einen gültigen Sympy-Ausdruck in t, z. B. exp(-2*t).\n\n"
            "Wie berechne ich das:\n"
            "- Es wird die Laplace-Transformation L{f(t)} gebildet und vereinfacht.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- F(s) beschreibt das System im s-Bereich und ist die Basis für Stabilitäts- und Regleranalysen."
        ),
        "Inverse Laplace": (
            "Was suche ich:\n"
            "- Die Zeitfunktion f(t) zu einer Bildfunktion F(s).\n\n"
            "Was brauche ich:\n"
            "- Einen gültigen Sympy-Ausdruck in s, z. B. 1/(s+3).\n\n"
            "Wie berechne ich das:\n"
            "- Es wird die inverse Laplace-Transformation L^{-1}{F(s)} durchgeführt.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- f(t) zeigt das Zeitverhalten des Systems direkt."
        ),
        "Partialbruchzerlegung": (
            "Was suche ich:\n"
            "- Die Zerlegung einer rationalen Funktion in einfache Teilbrüche.\n\n"
            "Was brauche ich:\n"
            "- Zähler- und Nennerkoeffizienten als Kommaliste.\n\n"
            "Wie berechne ich das:\n"
            "- Aus den Polynomkoeffizienten wird G(s) aufgebaut und per apart zerlegt.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- Die Teilbrüche erleichtern die Rücktransformation und Systeminterpretation."
        ),
        "Hurwitz-Kriterium": (
            "Was suche ich:\n"
            "- Ob das charakteristische Polynom stabil ist.\n\n"
            "Was brauche ich:\n"
            "- Nennerkoeffizienten des Polynoms.\n\n"
            "Wie berechne ich das:\n"
            "- Die Hurwitz-Matrix und ihre Hauptminoren werden bestimmt.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- Sind alle Hauptminoren positiv, ist das System stabil."
        ),
        "Routh-Kriterium": (
            "Was suche ich:\n"
            "- Die Anzahl instabiler Pole und die Stabilität.\n\n"
            "Was brauche ich:\n"
            "- Nennerkoeffizienten des charakteristischen Polynoms.\n\n"
            "Wie berechne ich das:\n"
            "- Das Routh-Schema wird aufgebaut, anschließend Vorzeichenwechsel in Spalte 1 gezählt.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- Keine Vorzeichenwechsel bedeuten stabile Pole in der linken Halbebene."
        ),
        "Reihenschaltung": (
            "Was suche ich:\n"
            "- Die Gesamtübertragungsfunktion zweier hintereinandergeschalteter Systeme.\n\n"
            "Was brauche ich:\n"
            "- Zähler und Nenner für G1(s) und G2(s).\n\n"
            "Wie berechne ich das:\n"
            "- Zähler und Nenner werden jeweils multipliziert.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- Die neue G(s) beschreibt das kombinierte Verhalten beider Teilsysteme."
        ),
        "Parallelschaltung": (
            "Was suche ich:\n"
            "- Die Gesamtübertragungsfunktion zweier parallelgeschalteter Systeme.\n\n"
            "Was brauche ich:\n"
            "- Zähler und Nenner für G1(s) und G2(s).\n\n"
            "Wie berechne ich das:\n"
            "- Es wird auf gemeinsamen Nenner erweitert und addiert.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- Die resultierende G(s) zeigt den kombinierten parallelen Signalweg."
        ),
        "Rueckkopplung": (
            "Was suche ich:\n"
            "- Die geschlossene Übertragungsfunktion eines Regelkreises.\n\n"
            "Was brauche ich:\n"
            "- Vorwärtszweig Gv(s), Rückführzweig Gr(s), sowie negativ/positiv.\n\n"
            "Wie berechne ich das:\n"
            "- Der geschlossene Nenner wird mit dem offenen Kreis und Vorzeichen gebildet.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- Du siehst direkt, wie die Rückkopplung die Dynamik verändert."
        ),
        "Sprungfaehigkeit": (
            "Was suche ich:\n"
            "- Ob das System sprunfähig und kausal realisierbar ist.\n\n"
            "Was brauche ich:\n"
            "- Zähler- und Nennerkoeffizienten der Übertragungsfunktion.\n\n"
            "Wie berechne ich das:\n"
            "- Die Grade von Zähler und Nenner werden verglichen.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- Sprungfähigkeit und Realisierbarkeit geben die praktische Umsetzbarkeit an."
        ),
        "Sprungantwort (G(s), F_ex)": (
            "Was suche ich:\n"
            "- Die Ausgangsantwort y(t) auf einen Einheitssprung bei gegebener G(s) und F_ex(s).\n\n"
            "Was brauche ich:\n"
            "- G(s), optional F_ex(s), und einen Endzeitwert für numerische Darstellung.\n\n"
            "Wie berechne ich das:\n"
            "- Es wird G_gesamt(s)=G(s)*F_ex(s) gebildet und über G_gesamt(s)/s ausgewertet.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- Du erhältst das Zeitverhalten; bei Symbolik zusätzlich eine analytische y(t)-Form."
        ),
        "Reglerentwurf (ZN/CC)": (
            "Was suche ich:\n"
            "- Startparameter für P-, PI- oder PID-Regler.\n\n"
            "Was brauche ich:\n"
            "- Reglertyp, Verfahren (ZN/CC), Modus (offen/geschlossen) und die zugehörigen Messwerte.\n"
            "- Offen: K, T, K_T; geschlossen (ZN): K_krit, T_krit.\n\n"
            "Wie berechne ich das:\n"
            "- Die Tabellenformeln des gewählten Verfahrens werden direkt angewendet.\n"
            "- Gültigkeit wird geprüft (ZN: K_T<0.5T, CC: K_T<2T).\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- Kp, Ki, Kd (sowie Ti/Td) sind sinnvolle Startwerte für die weitere Feinabstimmung."
        ),
        "Wurzelortsauslegung": (
            "Was suche ich:\n"
            "- Geeignete Verstärkungen k und das Polverhalten des geschlossenen Kreises.\n\n"
            "Was brauche ich:\n"
            "- Zähler/Nenner, k-Bereich sowie optional Dämpfungs- und Sigma-Grenzen.\n\n"
            "Wie berechne ich das:\n"
            "- Für den Bereich wird den(s)+k*num(s)=0 gelöst, Pole und Kennwerte werden tabelliert.\n\n"
            "Was sagt mir mein Ergebnis:\n"
            "- Empfohlene k-Werte und Pole zeigen, welche Einstellungen stabil und sinnvoll sind."
        ),
    }
    return explanations.get(
        name,
        (
            "Was suche ich:\n- Eine klare Zielgröße der ausgewählten Berechnung.\n\n"
            "Was brauche ich:\n- Die Eingabeparameter aus den Feldern links.\n\n"
            "Wie berechne ich das:\n- Das Tool nutzt die passende Regelungstechnik-Methode automatisch.\n\n"
            "Was sagt mir mein Ergebnis:\n- Der Output zeigt dir die berechnete Zielgröße und den Lösungsweg."
        ),
    )


class RegelungstechnikGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Regelungstechnik Toolkit")
        self.root.geometry("980x720")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(sticky="nsew")

        self.calculation_frame = ttk.Frame(self.notebook)
        self.plot_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.calculation_frame, text="Berechnungen")
        self.notebook.add(self.plot_frame, text="Plots")

        self.current_steps: List[Any] = []
        self.current_step_index = 0
        self.plot_history: List[Dict[str, Any]] = []
        self.plot_preview_image: tk.PhotoImage | None = None

        self._setup_calculation_tab()
        self._setup_plot_tab()

    def _setup_calculation_tab(self) -> None:
        self.calculation_frame.columnconfigure(0, weight=3)
        self.calculation_frame.columnconfigure(1, weight=1)
        self.calculation_frame.rowconfigure(3, weight=0)
        self.calculation_frame.rowconfigure(4, weight=0)
        self.calculation_frame.rowconfigure(5, weight=1)

        self.calc_functions: Dict[str, FunctionSpec] = {
            "Laplace-Transformation": {
                "func": laplace_transform,
                "fields": [("Zeitfunktion f(t)", "exp(-2*t)")],
                "help": "Sympy-Ausdruck in t eingeben, z.B. exp(-2*t)"},
            "Inverse Laplace": {
                "func": inverse_laplace,
                "fields": [("F(s)", "1/(s+3)")],
                "help": "Sympy-Ausdruck in s eingeben, z.B. 1/(s+3)"},
            "Partialbruchzerlegung": {
                "func": partialbruchzerlegung,
                "fields": [("Zaehler", "1, 2"), ("Nenner", "1, 3, 2")],
                "help": "Koeffizienten als Komma-liste eingeben"},
            "Hurwitz-Kriterium": {
                "func": hurwitz_kriterium,
                "fields": [("Nenner", "1, 5, 6")],
                "help": "Charakteristisches Polynom coefficients"},
            "Routh-Kriterium": {
                "func": routh_kriterium,
                "fields": [("Nenner", "1, 5, 6")],
                "help": "Charakteristisches Polynom coefficients"},
            "Reihenschaltung": {
                "func": reihenschaltung,
                "fields": [("G1 Zaehler", "1"), ("G1 Nenner", "1, 1"),
                           ("G2 Zaehler", "1"), ("G2 Nenner", "1, 2")],
                "help": "Zwei Uebertragungsfunktionen eingeben"},
            "Parallelschaltung": {
                "func": parallelschaltung,
                "fields": [("G1 Zaehler", "1"), ("G1 Nenner", "1, 1"),
                           ("G2 Zaehler", "1"), ("G2 Nenner", "1, 2")],
                "help": "Zwei Uebertragungsfunktionen eingeben"},
            "Rueckkopplung": {
                "func": rueckkopplung,
                "fields": [("Gv Zaehler", "1"), ("Gv Nenner", "1, 1"),
                           ("Gr Zaehler", "1"), ("Gr Nenner", "1, 2"),
                           ("Negativ", "True")],
                "help": "Geben Sie Vorwaerts- und Rueckfuehrungszweig ein"},
            "Sprungfaehigkeit": {
                "func": sprungfaehigkeit_realisierbarkeit,
                "fields": [("Zaehler", "1, 2"), ("Nenner", "1, 3, 2")],
                "help": "Uebertragungsfunktion fuer Sprungfaehigkeit"},
            "Sprungantwort (G(s), F_ex)": {
                "func": sprungantwort_mit_fex,
                "fields": [("G(s)", "1/(s+1)"), ("F_ex", "1"), ("T Endwert", "10")],
                "help": "Geben Sie G(s) und F_ex als Terme ein, z.B. 1/(s+1) oder 1/(s^2+2*s+1)"},
            "Reglerentwurf (ZN/CC)": {
                "func": reglerparameter_nach_verfahren,
                "fields": [("Reglertyp (P/PI/PID)", "PID"), ("Verfahren (ziegler-nichols/cohen-coon)", "ziegler-nichols"),
                           ("Modus (offen/geschlossen)", "offen"), ("K", "1.0"), ("T", "1.0"),
                           ("K_T", "0.2"), ("K_krit (optional)", ""), ("T_krit (optional)", "")],
                "help": "ZN offen: K,T,K_T mit K_T<0.5T; ZN geschlossen: K_krit,T_krit; CC: K,T,K_T mit K_T<2T"},
            "Wurzelortsauslegung": {
                "func": wurzelortsauslegung,
                "fields": [("Zaehler", "1"), ("Nenner", "1, 3, 2"), ("k start", "0"),
                           ("k ende", "20"), ("Anzahl k", "81"), ("Daempfung min (optional)", "0.4"),
                           ("Sigma Grenze (optional)", "-0.1")],
                "help": "Berechnet Pole-Tabelle, sinnvolle k-Werte und Asymptoten für die Wurzelortsauslegung"},
        }

        self.calc_selection = tk.StringVar(value="Laplace-Transformation")
        selection_frame = ttk.Frame(self.calculation_frame)
        selection_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        selection_frame.columnconfigure(1, weight=1)

        ttk.Label(selection_frame, text="Funktion:").grid(row=0, column=0, sticky="w")
        functions = list(self.calc_functions.keys())
        self.calc_dropdown = ttk.Combobox(selection_frame, values=functions,
                                          textvariable=self.calc_selection,
                                          state="readonly")
        self.calc_dropdown.grid(row=0, column=1, sticky="ew", padx=10)
        self.calc_dropdown.bind("<<ComboboxSelected>>", lambda _: self._render_calc_fields())

        self.calc_help_label = ttk.Label(selection_frame, text="", foreground="gray")
        self.calc_help_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

        workflow_frame = ttk.LabelFrame(self.calculation_frame, text="Leitfaden Zur Gewählten Berechnung")
        workflow_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)
        workflow_frame.columnconfigure(0, weight=1)
        workflow_frame.rowconfigure(0, weight=1)
        self.workflow_guide_text = ScrolledText(workflow_frame, wrap="word", height=18)
        self.workflow_guide_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.workflow_guide_text.configure(state="disabled")

        self.field_frame = ttk.Frame(self.calculation_frame)
        self.field_frame.grid(row=1, column=0, sticky="ew", padx=10)
        self.field_frame.columnconfigure(1, weight=1)

        self.calc_entries: Dict[str, tk.Widget] = {}
        self._render_calc_fields()

        button_frame = ttk.Frame(self.calculation_frame)
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.want_plot_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(button_frame, text="Plot erzeugen?", variable=self.want_plot_var).pack(anchor="w")
        ttk.Button(button_frame, text="Berechnen", command=self._run_calculation).pack(anchor="w")
        ttk.Button(button_frame, text="WolframAlpha Query kopieren", command=self._copy_wolframalpha_query).pack(anchor="w", pady=(4, 0))

        self.output_text = ScrolledText(self.calculation_frame, wrap="word", height=10)
        self.output_text.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        self.output_text.configure(state="disabled")

        step_nav_frame = ttk.Frame(self.calculation_frame)
        step_nav_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        step_nav_frame.columnconfigure(1, weight=1)
        self.prev_step_button = ttk.Button(step_nav_frame, text="< Vorheriger Schritt", command=self._prev_step)
        self.prev_step_button.grid(row=0, column=0, sticky="w")
        self.step_status_label = ttk.Label(step_nav_frame, text="Schritt 0/0")
        self.step_status_label.grid(row=0, column=1, sticky="ew")
        self.next_step_button = ttk.Button(step_nav_frame, text="Nächster Schritt >", command=self._next_step)
        self.next_step_button.grid(row=0, column=2, sticky="e")

        self.step_text = ScrolledText(self.calculation_frame, wrap="word", height=8)
        self.step_text.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        self.step_text.configure(state="disabled")

        self._update_step_navigation()

    def _setup_plot_tab(self) -> None:
        self.plot_frame.columnconfigure(0, weight=3)
        self.plot_frame.columnconfigure(1, weight=2)
        self.plot_frame.rowconfigure(3, weight=1)
        self.plot_frame.rowconfigure(4, weight=2)

        self.plot_functions: Dict[str, FunctionSpec] = {
            "Bode-Diagramm": {
                "func": plot_bode,
                "fields": [("Zaehler", "1, 2"), ("Nenner", "1, 3, 2")],
                "help": "Plot fuer Amplitude und Phase berechnen."},
            "Ortskurve": {
                "func": plot_ortskurve,
                "fields": [("Zaehler", "1, 2"), ("Nenner", "1, 3, 2")],
                "help": "Nyquist-Ortskurve der Uebertragungsfunktion."},
            "Wurzelortskurve": {
                "func": plot_wurzelortskurve,
                "fields": [("Zaehler", "1, 2"), ("Nenner", "1, 3, 2"),
                           ("k max", "10")],
                "help": "Trajektorien der Pole in Abhaengigkeit von k."},
            "Pol-Nullstellen-Diagramm": {
                "func": plot_pol_nullstellen,
                "fields": [("Zaehler", "1, 2"), ("Nenner", "1, 3, 2")],
                "help": "Pole und Nullstellen in der komplexen Ebene."},
            "Sprungantwort": {
                "func": plot_sprungantwort,
                "fields": [("Zaehler", "1"), ("Nenner", "1, 3, 2"),
                           ("T Endwert", "10")],
                "help": "Sprungantwort berechnen und speichern."},
        }

        self.plot_selection = tk.StringVar(value="Bode-Diagramm")
        plot_select_frame = ttk.Frame(self.plot_frame)
        plot_select_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        plot_select_frame.columnconfigure(1, weight=1)

        ttk.Label(plot_select_frame, text="Plottyp:").grid(row=0, column=0, sticky="w")
        self.plot_dropdown = ttk.Combobox(plot_select_frame,
                                         values=list(self.plot_functions.keys()),
                                         textvariable=self.plot_selection,
                                         state="readonly")
        self.plot_dropdown.grid(row=0, column=1, sticky="ew", padx=10)
        self.plot_dropdown.bind("<<ComboboxSelected>>", lambda _: self._render_plot_fields())

        self.plot_help_label = ttk.Label(plot_select_frame, text="", foreground="gray")
        self.plot_help_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

        self.plot_field_frame = ttk.Frame(self.plot_frame)
        self.plot_field_frame.grid(row=1, column=0, sticky="ew", padx=10)
        self.plot_field_frame.columnconfigure(1, weight=1)

        self.plot_entries: Dict[str, tk.Entry] = {}
        self._render_plot_fields()

        plot_button_frame = ttk.Frame(self.plot_frame)
        plot_button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        ttk.Button(plot_button_frame, text="Plot erzeugen", command=self._run_plot).pack(anchor="w")

        self.plot_output_text = ScrolledText(self.plot_frame, wrap="word", height=18)
        self.plot_output_text.grid(row=3, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.plot_output_text.configure(state="disabled")

        history_frame = ttk.LabelFrame(self.plot_frame, text="Plot-Liste")
        history_frame.grid(row=0, column=1, rowspan=5, sticky="nsew", padx=(0, 10), pady=10)
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=2)
        history_frame.rowconfigure(1, weight=1)
        history_frame.rowconfigure(2, weight=4)

        self.plot_history_list = tk.Listbox(history_frame, exportselection=False)
        self.plot_history_list.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 6))
        self.plot_history_list.bind("<<ListboxSelect>>", lambda _: self._show_selected_plot_history())

        self.plot_history_details = ScrolledText(history_frame, wrap="word", height=8)
        self.plot_history_details.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)
        self.plot_history_details.configure(state="disabled")

        preview_frame = ttk.LabelFrame(history_frame, text="Direkte Plot-Vorschau")
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(6, 8))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.plot_preview_label = ttk.Label(preview_frame, text="Noch kein Plot erzeugt.", anchor="center")
        self.plot_preview_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    def _clear_frame(self, frame: ttk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _render_calc_fields(self) -> None:
        self._clear_frame(self.field_frame)
        name = self.calc_selection.get()
        spec = self.calc_functions[name]
        self.calc_help_label.config(text=spec.get("help", ""))
        self.calc_entries = {}
        for idx, (label, default) in enumerate(spec["fields"]):
            ttk.Label(self.field_frame, text=label + ":").grid(row=idx, column=0, sticky="w", pady=4)
            if name == "Reglerentwurf (ZN/CC)" and label in {
                "Reglertyp (P/PI/PID)",
                "Verfahren (ziegler-nichols/cohen-coon)",
                "Modus (offen/geschlossen)",
            }:
                if label == "Reglertyp (P/PI/PID)":
                    values = ["P", "PI", "PID"]
                elif label == "Verfahren (ziegler-nichols/cohen-coon)":
                    values = ["ziegler-nichols", "cohen-coon"]
                else:
                    values = ["offen", "geschlossen"]
                combo = ttk.Combobox(self.field_frame, values=values, state="readonly")
                combo.grid(row=idx, column=1, sticky="ew", padx=5)
                combo.set(default)
                self.calc_entries[label] = combo
            else:
                entry = ttk.Entry(self.field_frame)
                entry.grid(row=idx, column=1, sticky="ew", padx=5)
                entry.insert(0, default)
                self.calc_entries[label] = entry
        self._update_workflow_guide(name)

    def _update_workflow_guide(self, calculation_name: str) -> None:
        text = _workflow_explanation(calculation_name)
        self.workflow_guide_text.configure(state="normal")
        self.workflow_guide_text.delete("1.0", tk.END)
        self.workflow_guide_text.insert(tk.END, text)
        self.workflow_guide_text.configure(state="disabled")

    def _get_calc_value(self, label: str) -> str:
        widget = self.calc_entries[label]
        getter = getattr(widget, "get", None)
        if getter is None:
            raise ValueError(f"Eingabefeld {label} besitzt keine get()-Methode.")
        return str(getter())

    def _build_wolframalpha_query(self) -> str:
        name = self.calc_selection.get()
        if name == "Reglerentwurf (ZN/CC)":
            reglertyp = self._get_calc_value("Reglertyp (P/PI/PID)").strip().upper()
            verfahren = self._get_calc_value("Verfahren (ziegler-nichols/cohen-coon)").strip().lower()
            modus = self._get_calc_value("Modus (offen/geschlossen)").strip().lower()
            if verfahren == "ziegler-nichols" and modus == "geschlossen":
                k_krit = self._get_calc_value("K_krit (optional)").strip() or "Kkrit"
                t_krit = self._get_calc_value("T_krit (optional)").strip() or "Tkrit"
                return f"{reglertyp} controller Ziegler Nichols closed loop Kcrit={k_krit} Tcrit={t_krit}"
            k = self._get_calc_value("K").strip() or "K"
            t = self._get_calc_value("T").strip() or "T"
            k_t = self._get_calc_value("K_T").strip() or "L"
            return f"{reglertyp} controller {verfahren} {modus} process gain={k} time constant={t} dead time={k_t}"

        if name == "Wurzelortsauslegung":
            num = self._get_calc_value("Zaehler").strip()
            den = self._get_calc_value("Nenner").strip()
            return f"root locus ({num})/({den})"

        if name == "Sprungantwort (G(s), F_ex)":
            g_expr = self._get_calc_value("G(s)").strip()
            f_expr = self._get_calc_value("F_ex").strip()
            return f"inverse laplace ({g_expr})*({f_expr})/s"

        return "control systems calculator"

    def _copy_wolframalpha_query(self) -> None:
        try:
            query = self._build_wolframalpha_query()
            self.root.clipboard_clear()
            self.root.clipboard_append(query)
            self.root.update_idletasks()
            messagebox.showinfo("WolframAlpha", f"Query kopiert:\n{query}")
        except Exception as exc:
            messagebox.showerror("WolframAlpha", f"Query konnte nicht erzeugt werden: {exc}")

    def _render_plot_fields(self) -> None:
        self._clear_frame(self.plot_field_frame)
        name = self.plot_selection.get()
        spec = self.plot_functions[name]
        self.plot_help_label.config(text=spec.get("help", ""))
        self.plot_entries = {}
        for idx, (label, default) in enumerate(spec["fields"]):
            ttk.Label(self.plot_field_frame, text=label + ":").grid(row=idx, column=0, sticky="w", pady=4)
            entry = ttk.Entry(self.plot_field_frame)
            entry.grid(row=idx, column=1, sticky="ew", padx=5)
            entry.insert(0, default)
            self.plot_entries[label] = entry

    def _show_step(self) -> None:
        self.step_text.configure(state="normal")
        self.step_text.delete("1.0", tk.END)
        step_count = len(self.current_steps)
        if step_count == 0:
            self.step_text.insert(tk.END, "Kein Schritt verfügbar. Führen Sie zuerst eine Berechnung aus.")
            self.step_status_label.config(text="Schritt 0/0")
            self.prev_step_button.state(["disabled"])
            self.next_step_button.state(["disabled"])
        else:
            current_step = self.current_steps[self.current_step_index]
            self.step_text.insert(tk.END, _format_step(current_step))
            self.step_status_label.config(text=f"Schritt {self.current_step_index + 1}/{step_count}")
            if self.current_step_index == 0:
                self.prev_step_button.state(["disabled"])
            else:
                self.prev_step_button.state(["!disabled"])
            if self.current_step_index >= step_count - 1:
                self.next_step_button.state(["disabled"])
            else:
                self.next_step_button.state(["!disabled"])
        self.step_text.configure(state="disabled")

    def _update_step_navigation(self) -> None:
        self._show_step()

    def _prev_step(self) -> None:
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self._show_step()

    def _next_step(self) -> None:
        if self.current_step_index < len(self.current_steps) - 1:
            self.current_step_index += 1
            self._show_step()

    def _run_calculation(self) -> None:
        name = self.calc_selection.get()
        spec = self.calc_functions[name]
        func = spec["func"]
        try:
            if name == "Laplace-Transformation":
                arg = _parse_expression(self._get_calc_value("Zeitfunktion f(t)"))
                result = func(arg)
            elif name == "Inverse Laplace":
                arg = _parse_expression(self._get_calc_value("F(s)"))
                result = func(arg)
            elif name == "Partialbruchzerlegung":
                num = _parse_polynom(self._get_calc_value("Zaehler"))
                den = _parse_polynom(self._get_calc_value("Nenner"))
                result = func(num, den)
            elif name in {"Hurwitz-Kriterium", "Routh-Kriterium"}:
                den = _parse_polynom(self._get_calc_value("Nenner"))
                result = func(den)
            elif name == "Sprungfaehigkeit":
                num = _parse_polynom(self._get_calc_value("Zaehler"))
                den = _parse_polynom(self._get_calc_value("Nenner"))
                result = func(num, den)
            elif name == "Sprungantwort (G(s), F_ex)":
                g_expr = self._get_calc_value("G(s)")
                f_expr = self._get_calc_value("F_ex")
                g_num, g_den = _parse_transfer_function_expression(g_expr)
                f_num, f_den = _parse_transfer_function_expression(f_expr)
                t_end = float(self._get_calc_value("T Endwert"))
                result = func((g_num, g_den), (f_num, f_den), t_ende=t_end)
                if self.want_plot_var.get():
                    try:
                        plot_num = _coerce_numeric_coeffs(g_num)
                        plot_den = _coerce_numeric_coeffs(g_den)
                        plot_result = plot_sprungantwort(plot_num, plot_den, t_ende=t_end,
                                                         dateiname="sprungantwort_gui.png")
                        hints = list(result.get("hinweise", []))
                        hints.append(f"Plot gespeichert unter {plot_result['plot_pfad']}")
                        result["hinweise"] = hints
                        result["plot_pfad"] = plot_result["plot_pfad"]
                    except Exception as plot_exc:
                        hints = list(result.get("hinweise", []))
                        hints.append(f"Plot konnte nicht erzeugt werden: {plot_exc}")
                        result["hinweise"] = hints
            elif name == "Reglerentwurf (ZN/CC)":
                reglertyp = self._get_calc_value("Reglertyp (P/PI/PID)").strip()
                verfahren = self._get_calc_value("Verfahren (ziegler-nichols/cohen-coon)").strip()
                modus = self._get_calc_value("Modus (offen/geschlossen)").strip()
                k = _parse_optional_float(self._get_calc_value("K"))
                t = _parse_optional_float(self._get_calc_value("T"))
                k_t = _parse_optional_float(self._get_calc_value("K_T"))
                k_krit = _parse_optional_float(self._get_calc_value("K_krit (optional)"))
                t_krit = _parse_optional_float(self._get_calc_value("T_krit (optional)"))
                result = func(reglertyp=reglertyp, verfahren=verfahren, modus=modus,
                              K=k, T=t, K_T=k_t, K_krit=k_krit, T_krit=t_krit)
            elif name == "Wurzelortsauslegung":
                num = _parse_polynom(self._get_calc_value("Zaehler"))
                den = _parse_polynom(self._get_calc_value("Nenner"))
                k_start = float(self._get_calc_value("k start"))
                k_ende = float(self._get_calc_value("k ende"))
                anzahl_k = int(float(self._get_calc_value("Anzahl k")))
                daempfung_min = _parse_optional_float(self._get_calc_value("Daempfung min (optional)"))
                sigma_grenze = _parse_optional_float(self._get_calc_value("Sigma Grenze (optional)"))
                result = func(num, den, k_start=k_start, k_ende=k_ende, anzahl_k=anzahl_k,
                              daempfung_min=daempfung_min, sigma_grenze=sigma_grenze)
                if self.want_plot_var.get():
                    try:
                        k_bereich = result["ergebnis"].get("k_werte", [])
                        k_mark = result["ergebnis"].get("k_empfohlen", [])
                        plot_result = plot_wurzelortskurve(
                            num, den,
                            k_bereich=k_bereich,
                            k_markierungen=k_mark,
                            sigma_grenze=sigma_grenze,
                            daempfung_min=daempfung_min,
                            dateiname="wurzelortsauslegung_gui.png",
                        )
                        hints = list(result.get("hinweise", []))
                        hints.append(f"Plot gespeichert unter {plot_result['plot_pfad']}")
                        result["hinweise"] = hints
                        result["plot_pfad"] = plot_result["plot_pfad"]
                    except Exception as plot_exc:
                        hints = list(result.get("hinweise", []))
                        hints.append(f"Plot konnte nicht erzeugt werden: {plot_exc}")
                        result["hinweise"] = hints
            elif name in {"Reihenschaltung", "Parallelschaltung"}:
                g1_num = _parse_polynom(self._get_calc_value("G1 Zaehler"))
                g1_den = _parse_polynom(self._get_calc_value("G1 Nenner"))
                g2_num = _parse_polynom(self._get_calc_value("G2 Zaehler"))
                g2_den = _parse_polynom(self._get_calc_value("G2 Nenner"))
                result = func((g1_num, g1_den), (g2_num, g2_den))
            elif name == "Rueckkopplung":
                g1_num = _parse_polynom(self._get_calc_value("Gv Zaehler"))
                g1_den = _parse_polynom(self._get_calc_value("Gv Nenner"))
                g2_num = _parse_polynom(self._get_calc_value("Gr Zaehler"))
                g2_den = _parse_polynom(self._get_calc_value("Gr Nenner"))
                neg = self._get_calc_value("Negativ").strip().lower() not in {"false", "0", "no"}
                result = func((g1_num, g1_den), (g2_num, g2_den), negativ=neg)
            else:
                raise ValueError("Unbekannte Funktion")
            self.current_steps = result.get("loesungsweg", []) or []
            self.current_step_index = 0
            self._write_output(self.output_text, result, guidance=_guidance_for_result(name, result))
            self._show_step()
        except Exception as exc:
            self.current_steps = []
            self.current_step_index = 0
            self._write_output(self.output_text, {"ergebnis": None, "loesungsweg": [f"Fehler: {exc}"], "plot_pfad": None})
            self._show_step()

    def _run_plot(self) -> None:
        name = self.plot_selection.get()
        spec = self.plot_functions[name]
        func = spec["func"]
        try:
            params_snapshot: Dict[str, str] = {k: v.get() for k, v in self.plot_entries.items()}
            if name == "Wurzelortskurve":
                num = _parse_polynom(self.plot_entries["Zaehler"].get())
                den = _parse_polynom(self.plot_entries["Nenner"].get())
                kmax = float(self.plot_entries["k max"].get())
                result = func(num, den, k_bereich=list(np.linspace(0.01, kmax, 500)))
            elif name == "Sprungantwort":
                num = _parse_polynom(self.plot_entries["Zaehler"].get())
                den = _parse_polynom(self.plot_entries["Nenner"].get())
                t_end = float(self.plot_entries["T Endwert"].get())
                result = func(num, den, t_ende=t_end)
            else:
                num = _parse_polynom(self.plot_entries["Zaehler"].get())
                den = _parse_polynom(self.plot_entries["Nenner"].get())
                result = func(num, den)
            self._write_output(self.plot_output_text, result)
            self._add_plot_history_entry(name, params_snapshot, result)
            self._display_plot_preview(result.get("plot_pfad"))
        except Exception as exc:
            self._write_output(self.plot_output_text, {"ergebnis": None, "loesungsweg": [f"Fehler: {exc}"], "plot_pfad": None})

    def _display_plot_preview(self, plot_path: str | None) -> None:
        if not plot_path:
            self.plot_preview_image = None
            self.plot_preview_label.configure(text="Kein Plotpfad vorhanden.", image="")
            return
        try:
            self.plot_preview_image = tk.PhotoImage(file=plot_path)
            self.plot_preview_label.configure(image=self.plot_preview_image, text="")
        except Exception as exc:
            self.plot_preview_image = None
            self.plot_preview_label.configure(
                text=f"Plot wurde gespeichert, Vorschau konnte nicht geladen werden:\n{plot_path}\n{exc}",
                image="",
            )

    def _add_plot_history_entry(self, plot_name: str, params: Dict[str, str], result: Dict[str, Any]) -> None:
        entry = {
            "name": plot_name,
            "params": dict(params),
            "plot_pfad": result.get("plot_pfad"),
            "loesungsweg": list(result.get("loesungsweg", [])),
        }
        self.plot_history.append(entry)
        short_params = ", ".join(f"{k}={v}" for k, v in params.items())
        label = f"{plot_name} | {short_params}" if short_params else plot_name
        self.plot_history_list.insert(tk.END, label)
        last_index = self.plot_history_list.size() - 1
        self.plot_history_list.selection_clear(0, tk.END)
        self.plot_history_list.selection_set(last_index)
        self.plot_history_list.see(last_index)
        self._show_selected_plot_history()

    def _show_selected_plot_history(self) -> None:
        selection = self.plot_history_list.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx < 0 or idx >= len(self.plot_history):
            return
        entry = self.plot_history[idx]
        lines: List[str] = []
        lines.append(f"Plottyp: {entry['name']}")
        params = entry.get("params", {})
        if params:
            lines.append("Parameter:")
            for key, value in params.items():
                lines.append(f"- {key}: {value}")
        if entry.get("plot_pfad"):
            lines.append(f"Pfad: {entry['plot_pfad']}")
        loesungsweg = entry.get("loesungsweg", [])
        if loesungsweg:
            lines.append("Beschreibung:")
            for step in loesungsweg[:3]:
                lines.append(f"- {_format_step(step)}")

        self.plot_history_details.configure(state="normal")
        self.plot_history_details.delete("1.0", tk.END)
        self.plot_history_details.insert(tk.END, "\n".join(lines))
        self.plot_history_details.configure(state="disabled")

        self._display_plot_preview(entry.get("plot_pfad"))

    def _write_output(self, widget: ScrolledText, result: Dict[str, Any], guidance: str | None = None) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, _build_output(result))
        if guidance:
            widget.insert(tk.END, f"\n\nNächste Schritte:\n{guidance}")
        widget.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    app = RegelungstechnikGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
