import tkinter as tk

import pytest

from regelungstechnik.gui import RegelungstechnikGUI, _build_output


def _safe_root():
    try:
        root = tk.Tk()
        root.withdraw()
        return root
    except tk.TclError:
        pytest.skip("Tkinter GUI ist in dieser Umgebung nicht verfügbar")


def test_build_output_shows_error_details():
    result = {"ergebnis": None, "loesungsweg": ["Fehler: Testfehler"], "plot_pfad": None}
    text = _build_output(result)
    assert "Fehler:" in text
    assert "Testfehler" in text


def test_gui_pipeline_laplace_produces_result_text():
    root = _safe_root()
    try:
        app = RegelungstechnikGUI(root)
        app.calc_selection.set("Laplace-Transformation")
        app._render_calc_fields()
        app.calc_entries["Zeitfunktion f(t)"].delete(0, tk.END)
        app.calc_entries["Zeitfunktion f(t)"].insert(0, "exp(-2*t)")

        app._run_calculation()

        app.output_text.configure(state="normal")
        content = app.output_text.get("1.0", tk.END)
        app.output_text.configure(state="disabled")
        assert "None" not in content
        assert "1/(s + 2)" in content or "1/(s+2)" in content
    finally:
        root.destroy()


def test_gui_pipeline_sprungfaehigkeit_produces_result_text():
    root = _safe_root()
    try:
        app = RegelungstechnikGUI(root)
        app.calc_selection.set("Sprungfaehigkeit")
        app._render_calc_fields()
        app.calc_entries["Zaehler"].delete(0, tk.END)
        app.calc_entries["Zaehler"].insert(0, "1, 2")
        app.calc_entries["Nenner"].delete(0, tk.END)
        app.calc_entries["Nenner"].insert(0, "1, 3, 2")

        app._run_calculation()

        app.output_text.configure(state="normal")
        content = app.output_text.get("1.0", tk.END)
        app.output_text.configure(state="disabled")
        assert "sprungfaehig" in content.lower()
        assert "None" not in content
    finally:
        root.destroy()
