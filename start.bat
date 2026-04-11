@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo ===========================================================
echo    HFSS-Python-Optimizer
echo ===========================================================
echo.

REM 显示 Python 路径
echo Checking Python environment...
where python
echo.

REM 显示 Python 版本
python --version
echo.

REM 检查 pyaedt
echo Checking pyaedt...
python -c "import pyaedt; print('pyaedt version:', pyaedt.__version__)" 2>nul
if errorlevel 1 (
    echo [ERROR] pyaedt not found in this Python environment!
    echo.
    echo Please install pyaedt first:
    echo   pip install pyaedt
    echo.
    echo Or use the correct Python path to install:
    echo   where python
    echo   "path_to_python" -m pip install pyaedt
    echo.
    pause
    exit /b 1
)
echo.

REM 启动程序
echo Starting HFSS-Python-Optimizer...
python "%~dp0launch_gui.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start. Please check Python and dependencies.
    echo Install dependencies: pip install -r requirements.txt
    pause
)
