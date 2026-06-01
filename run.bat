@echo off
REM Launch the OpenCV Prototyping Tool using the project's virtual environment.
REM Any arguments (e.g. an image path) are forwarded to the app.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run] .venv not found in "%cd%".
    echo [run] Create it once with:
    echo         py -3.13 -m venv .venv
    echo         .venv\Scripts\python -m pip install -r requirements.txt
    exit /b 1
)

".venv\Scripts\python.exe" main.py %*
