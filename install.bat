@echo off
echo Setting up k-line-predict-mini environment...

REM Check if Python 3 is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3 is not installed. Please install Python 3.8 or higher.
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Found Python version: %PYTHON_VERSION%

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install package in development mode
echo Installing package in development mode...
pip install -e .

echo.
echo Installation complete!
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the stock predictor, use:
echo   python main.py ^<SYMBOL^> ^<MARKET^>
echo   or
echo   stock-predict ^<SYMBOL^> ^<MARKET^>
echo.
echo To deactivate the virtual environment, run:
echo   deactivate