@echo off
cd /d "%~dp0"
python dashboard\main.py
if errorlevel 1 pause
