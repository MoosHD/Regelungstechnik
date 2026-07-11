from pathlib import Path
from sys import executable
from subprocess import run
text = ('[project]\n'
        'name = "regelungstechnik"\n'
        'version = "0.1.0"\n'
        'description = "Regelungstechnik Toolkit fuer Laplace, Stabilitaet, Reglerentwurf und Plotterstellung."\n'
        'readme = "README.md"\n'
        'requires-python = ">=3.9"\n'
        'dependencies = [\n'
        '  "numpy",\n'
        '  "scipy",\n'
        '  "matplotlib",\n'
        '  "pandas",\n'
        '  "sympy",\n'
        '  "click"\n'
        ']\n\n'
        '[build-system]\n'
        'requires = ["setuptools>=61.0", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n')
Path('pyproject.toml').write_text(text, encoding='utf-8')
run([executable, '-m', 'pytest', '-q'], check=True)
