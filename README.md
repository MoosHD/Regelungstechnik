# Regelungstechnik Skript

Dieses Repository enthaelt ein Python-Paket fuer Regelungstechnik mit Modulen zu Laplace, Stabilitaet, Reglersynthese und Plots.

## Schnellstart (Notebook-basiert)

Fuer eine schnelle Anwendung im Vorlesungsablauf ist das Notebook der Hauptworkflow:

- `regelungstechnik_workflow.ipynb`

Reihenfolge im Notebook:

1. Laplace / Inverse Laplace / Partialbruch
2. Blockschaltbildrechnung
3. Stabilitaet (Hurwitz, Routh, allgemeines Nyquist)
4. Entwurf (Wurzelort, ZN/CC, Phasenkorrekturglied)
5. Verifikation (Bode, Nyquist, Sprungantwort)

## Installation

```bash
python -m pip install -r REQUIREMENTS.txt
```

## Nutzung

```bash
python -m regelungstechnik --help
```

Wichtige neue Funktionen:

- Allgemeines Nyquist-Kriterium inkl. Stabilitaetsentscheidung (`nyquist_kriterium`)
- Nyquist-Plot (`plot_nyquist`)
- Auslegung phasenanhebendes/phasenabsenkendes Korrekturglied (`phasenkorrekturglied_auslegung`)

## Tests

```bash
pytest tests
```
