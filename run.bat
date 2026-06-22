@echo off
cd /d "%~dp0"
set "PY311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
rem MediaPipe on Windows cannot load model resources from a path containing
rem non-ASCII characters. Use a temporary ASCII-only drive mapping.
subst R: "%~dp0." >nul 2>&1
if exist "R:\.venv\Lib\site-packages" set "PYTHONPATH=R:\.venv\Lib\site-packages"
if exist "%PY311%" (
    pushd R:\
    "%PY311%" main.py
    popd
) else (
    pushd R:\
    py -3.11 main.py
    popd
)
subst R: /d >nul 2>&1
if errorlevel 1 pause
