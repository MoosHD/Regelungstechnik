"""CLI fuer das Regelungstechnik-Paket."""
from __future__ import annotations
import click
import numpy as np
from sympy import sympify
from regelungstechnik.laplace import inverse_laplace, laplace_transform, partialbruchzerlegung
from regelungstechnik.plots import plot_bode, plot_ortskurve, plot_pol_nullstellen, plot_wurzelortskurve
from regelungstechnik.stabilitaet import hurwitz_kriterium, routh_kriterium
from regelungstechnik.reglerentwurf import (parallelschaltung, reihenschaltung,
                                           rueckkopplung, sprungfaehigkeit_realisierbarkeit)


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
@click.option('--kmax', default=10.0, show_default=True, help='Maximale Verstärkung.')
@click.option('--num', required=True, help='Zaehlerkoeffizienten, Komma-getrennt.')
@click.option('--den', required=True, help='Nennerkoeffizienten, Komma-getrennt.')
def wurzelort(num: str, den: str, kmax: float) -> None:
    result = plot_wurzelortskurve(_parse_polynom(num), _parse_polynom(den),
                                  k_bereich=list(np.linspace(0.01, kmax, 500)))
    click.echo(f"Wurzelortskurve gespeichert in: {result['plot_pfad']}")
