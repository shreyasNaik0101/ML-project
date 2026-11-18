@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The project environment is missing. Follow the setup steps in README.md first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m jupyterlab
if errorlevel 1 pause
