@echo off
REM Launches the MLflow tracking UI against ./mlruns.
REM Once started, open http://127.0.0.1:5000 in a browser.
REM Stop with Ctrl+C in this window.

setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run scripts\setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m mlflow ui --backend-store-uri file:./mlruns
