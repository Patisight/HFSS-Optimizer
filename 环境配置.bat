@echo off
chcp 65001 >nul 2>&1

:menu
cls
echo.
echo ===========================================================
echo    HFSS Antenna Optimizer - Setup
echo ===========================================================
echo.
echo  1. Check Environment
echo  2. Setup Environment
echo  3. Launch Program
echo  4. Exit
echo.

set /p choice=Enter option (1-4): 

if "%choice%"=="1" goto check
if "%choice%"=="2" goto setup
if "%choice%"=="3" goto launch
if "%choice%"=="4" goto end
goto menu

:check
cls
echo.
echo ===========================================================
echo    Environment Check
echo ===========================================================
python "%~dp0setup_env.py" --check
echo.
pause
goto menu

:setup
cls
echo.
echo ===========================================================
echo    Setup Environment
echo ===========================================================
python "%~dp0setup_env.py"
echo.
pause
goto menu

:launch
cls
cd /d "%~dp0"
start "" pythonw launch_gui.py
goto end

:end
