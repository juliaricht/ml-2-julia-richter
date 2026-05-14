@echo off
REM Launches JupyterLab from the project's venv.
REM Open notebooks/01_data_exploration.ipynb or notebooks/02_modeling.ipynb in the browser tab that opens.

setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run scripts\setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m jupyter lab
