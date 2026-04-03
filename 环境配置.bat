@echo off
REM ===========================================================
REM    HFSS 天线优化程序 - 环境配置工具
REM ===========================================================
chcp 65001 >nul 2>&1

:menu
cls
echo.
echo ===========================================================
echo    HFSS Antenna Optimizer - Setup
echo ===========================================================
echo.
echo  1. Check Environment
echo  2. Setup Environment (Basic)
echo  3. Setup Environment (Full - with GPflow)
echo  4. Install Surrogate Dependencies (GPflow)
echo  5. Launch Program
echo  6. Exit
echo.

set /p choice=Enter option (1-6): 

if "%choice%"=="1" goto check
if "%choice%"=="2" goto setup
if "%choice%"=="3" goto setup_full
if "%choice%"=="4" goto setup_surrogate
if "%choice%"=="5" goto launch
if "%choice%"=="6" goto end
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
echo    Setup Environment (Basic)
echo ===========================================================
echo.
echo This will install basic packages:
echo   - numpy, pandas, scipy
echo   - pyaedt (HFSS interface)
echo   - scikit-optimize, scikit-learn
echo   - matplotlib, PyQt6
echo.
set /p confirm=Continue? (Y/N): 
if /i not "%confirm%"=="Y" goto menu

python "%~dp0setup_env.py"
echo.
pause
goto menu

:setup_full
cls
echo.
echo ===========================================================
echo    Setup Environment (Full)
echo ===========================================================
echo.
echo This will install ALL packages including:
echo   - Basic packages (numpy, pandas, scipy, pyaedt, etc.)
echo   - GPflow + TensorFlow (for gpflow_svgp surrogate model)
echo.
echo NOTE: GPflow/TensorFlow is large (~500MB+), 
echo       only needed if you want to use gpflow_svgp model.
echo.
set /p confirm=Continue? (Y/N): 
if /i not "%confirm%"=="Y" goto menu

python "%~dp0setup_env.py" --full
echo.
pause
goto menu

:setup_surrogate
cls
echo.
echo ===========================================================
echo    Install Surrogate Dependencies
echo ===========================================================
echo.
echo This will install GPflow and TensorFlow for:
echo   - gpflow_svgp: Sparse Variational GP (recommended)
echo   - incremental: RFF+SGD incremental learning
echo.
echo NOTE: These packages are large (~500MB+ total).
echo.
set /p confirm=Continue? (Y/N): 
if /i not "%confirm%"=="Y" goto menu

python "%~dp0setup_env.py" --surrogate
echo.
pause
goto menu

:launch
cls
cd /d "%~dp0"
start "" pythonw "%~dp0launch_gui.py"
goto end

:end
