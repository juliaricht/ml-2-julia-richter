@echo off
REM Re-executes notebooks\02_modeling.ipynb top-to-bottom and saves the outputs back into the file.
REM Useful after changing src\modeling.py or hyperparameters.

setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run scripts\setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m jupyter nbconvert ^
    --to notebook ^
    --execute ^
    --inplace ^
    --ExecutePreprocessor.timeout=600 ^
    notebooks\02_modeling.ipynb

if errorlevel 1 (
    echo Notebook execution failed.
    pause
    exit /b 1
)

echo.
echo Notebook re-executed successfully.
pause
