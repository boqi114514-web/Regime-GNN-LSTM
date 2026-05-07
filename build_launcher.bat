@echo off
cd /d "%~dp0"
echo Current dir: %CD%
if exist "venv\Scripts\python.exe" goto :run
echo ERROR: venv\Scripts\python.exe not found
echo Run first: python -m venv venv
echo Then:      venv\Scripts\pip install -r requirements.txt
pause
exit /b 1
:run
venv\Scripts\python build_launcher.py
pause
