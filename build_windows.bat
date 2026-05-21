@echo off
chcp 65001 >nul
echo ============================================
echo   Vocal Pitch Monitor - Windows Build
echo ============================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv vpm_env
call vpm_env\Scripts\activate.bat

echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [3/5] Installing dependencies...
pip install -r requirements.txt --quiet

echo [4/5] Installing PyInstaller...
pip install pyinstaller --quiet

echo [5/5] Building executable...
pyinstaller --onefile --windowed --icon=assets\logo.ico --name="Vocal Pitch Monitor" --add-data "assets;assets" main.py

echo.
echo ============================================
echo   Build Complete!
echo   Executable: dist\Vocal Pitch Monitor.exe
echo ============================================
pause
