@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    set "PYTHON=.venv\Scripts\pythonw.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    where pythonw.exe >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON=pythonw"
    ) else (
        set "PYTHON=python"
    )
)

start "" "%PYTHON%" -m regelungstechnik gui

if errorlevel 1 (
    echo.
    echo Start fehlgeschlagen. Bitte Python-Umgebung pruefen.
    pause
)
