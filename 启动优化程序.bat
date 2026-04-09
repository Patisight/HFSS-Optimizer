@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
python "%~dp0launch_gui.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start GUI. Check error above.
    pause
)
