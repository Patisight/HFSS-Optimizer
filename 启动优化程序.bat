@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
start "" pythonw "%~dp0launch_gui.py"
