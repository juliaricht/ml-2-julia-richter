@echo off
REM Installs the CUDA build of PyTorch into .venv.
REM Run BEFORE scripts\setup.bat (so requirements.txt sees torch as already satisfied).
REM Requires an NVIDIA GPU with a recent driver. Without a GPU, run setup.bat directly
REM and it will install the CPU build of torch from PyPI.

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

echo Installing CUDA build of torch (cu124 index) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 --disable-pip-version-check

if errorlevel 1 (
    echo Install failed.
    pause
    exit /b 1
)

echo.
echo Done. Now run scripts\setup.bat to install the rest of requirements.txt.
pause
