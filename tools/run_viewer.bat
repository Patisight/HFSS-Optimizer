@echo off
cd /d "%~dp0.."
echo ========================================
echo    HFSS Result Viewer
echo ========================================
echo.
python tools\result_viewer.py
if errorlevel 1 (
    echo.
    echo Error occurred. Check Python environment.
    pause
)
