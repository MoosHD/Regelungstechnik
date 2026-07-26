"""CLI fuer das Regelungstechnik-Paket."""
from __future__ import annotations
import click
import numpy as np
from sympy import sympify
from regelungstechnik.laplace import inverse_laplace, laplace_transform, partialbruchzerlegung
from regelungstechnik.plots import (plot_bode, plot_nyquist, plot_ortskurve,
                                    plot_pol_nullstellen, plot_wurzelortskurve)
from regelungstechnik.stabilitaet import hurwitz_kriterium, nyquist_kriterium, routh_kriterium
from regelungstechnik.reglerentwurf import (parallelschaltung, reihenschaltung,
                                           rueckkopplung, sprungfaehigkeit_realisierbarkeit,
                                           phasenkorrekturglied_auslegung)


def _parse_polynom(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(',') if x.strip()]


@click.group()
def cli() -> None:
    """Regelungstechnik CLI fuer schnelle Berechnungen und Plots."""


@cli.command()
@click.argument('expression')
def laplace(expression: str) -> None:
    expr = sympify(expression)
    result = laplace_transform(expr)
    click.echo(result['ergebnis'])


@cli.command()
@click.argument('expression')
def ilaplace(expression: str) -> None:
    expr = sympify(expression)
    result = inverse_laplace(expr)
    click.echo(result['ergebnis'])


@cli.command()
@click.option('--num', required=True, help='Zaehlerkoeffizienten, Komma-getrennt.')
@click.option('--den', required=True, help='Nennerkoeffizienten, Komma-getrennt.')
def partialbruch(num: str, den: str) -> None:
    result = partialbruchzerlegung(_parse_polynom(num), _parse_polynom(den))
    click.echo(result['ergebnis'])


@cli.command()
@click.option('--num', required=True, help='Zaehlerkoeffizienten, Komma-getrennt.')
@click.option('--den', required=True, help='Nennerkoeffizienten, Komma-getrennt.')
def bode(num: str, den: str) -> None:
    result = plot_bode(_parse_polynom(num), _parse_polynom(den))
    click.echo(f"Bode plot gespeichert in: {result['plot_pfad']}")


@cli.command()
@click.option('--num', required=True, help='Zaehlerkoeffizienten, Komma-getrennt.')
@click.option('--den', required=True, help='Nennerkoeffizienten, Komma-getrennt.')
def ortskurve(num: str, den: str) -> None:
    result = plot_ortskurve(_parse_polynom(num), _parse_polynom(den))
    click.echo(f"Ortskurve gespeichert in: {result['plot_pfad']}")


@cli.command()
@click.option('--den', required=True, help='Nennerkoeffizienten, Komma-getrennt.')
def routh(den: str) -> None:
    result = routh_kriterium(_parse_polynom(den))
    click.echo(result['ergebnis'])


@cli.command()
@click.option('--den', required=True, help='Nennerkoeffizienten, Komma-getrennt.')
def hurwitz(den: str) -> None:
    result = hurwitz_kriterium(_parse_polynom(den))
    click.echo(result['ergebnis'])


@cli.command()
@click.option('--num', required=True, help='Zaehlerkoeffizienten, Komma-getrennt.')
@click.option('--den', required=True, help='Nennerkoeffizienten, Komma-getrennt.')
@click.option('--wmin', default=1e-3, show_default=True, type=float)
@click.option('--wmax', default=1e3, show_default=True, type=float)
@click.option('--punkte', default=3000, show_default=True, type=int)
@click.option('--plot', is_flag=True, help='Nyquist-Plot speichern.')
def nyquist(num: str, den: str, wmin: float, wmax: float, punkte: int, plot: bool) -> None:
    num_vals = _parse_polynom(num)
    den_vals = _parse_polynom(den)
    result = nyquist_kriterium(num_vals, den_vals, w_min=wmin, w_max=wmax, punkte=punkte)
    click.echo(result['ergebnis'])
    if plot:
        plot_result = plot_nyquist(num_vals, den_vals, w_min=wmin, w_max=wmax, punkte=punkte)
        click.echo(f"Nyquist-Plot gespeichert in: {plot_result['plot_pfad']}")


@cli.command()
@click.option('--kmax', default=10.0, show_default=True, help='Maximale Verstärkung.')
@click.option('--num', required=True, help='Zaehlerkoeffizienten, Komma-getrennt.')
@click.option('--den', required=True, help='Nennerkoeffizienten, Komma-getrennt.')
def wurzelort(num: str, den: str, kmax: float) -> None:
    result = plot_wurzelortskurve(_parse_polynom(num), _parse_polynom(den),
                                  k_bereich=list(np.linspace(0.01, kmax, 500)))
    click.echo(f"Wurzelortskurve gespeichert in: {result['plot_pfad']}")


@cli.command()
@click.option('--typ', required=True, type=click.Choice(['anhebend', 'absenkend'], case_sensitive=False))
@click.option('--phi', required=True, type=float, help='Gewuenschter Betrag der Phasenkorrektur in Grad.')
@click.option('--omega', required=True, type=float, help='Auslegungs-Kreisfrequenz omega_c [rad/s].')
@click.option('--K', 'k_wert', default=1.0, show_default=True, type=float, help='Statische Verstaerkung des Korrekturglieds.')
def phasenkorrektur(typ: str, phi: float, omega: float, k_wert: float) -> None:
    result = phasenkorrekturglied_auslegung(typ=typ, phi_grad=phi, omega_c=omega, K=k_wert)
    click.echo(result['ergebnis'])
