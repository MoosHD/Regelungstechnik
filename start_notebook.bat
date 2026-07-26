@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" -m jupyter lab regelungstechnik_workflow.ipynb

if errorlevel 1 (
    echo.
    echo Start fehlgeschlagen. Bitte Python-Umgebung und Jupyter-Installation pruefen.
    pause
)
