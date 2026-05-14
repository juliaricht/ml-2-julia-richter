@echo off
REM Creates .venv if missing and installs / updates requirements.
REM Safe to run repeatedly.

setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    py -m venv .venv
    if errorlevel 1 (
        echo Failed to create venv. Is Python installed? Try: py --version
        pause
        exit /b 1
    )
)

echo Installing requirements ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check
".venv\Scripts\python.exe" -m pip install -r requirements.txt --disable-pip-version-check

if errorlevel 1 (
    echo Install failed.
    pause
    exit /b 1
)

echo.
echo Done. The venv lives in .venv\
pause
