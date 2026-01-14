@echo off
REM VAID JI Quick Setup Script for Windows
REM This script automates the setup process for VAID JI

echo ================================================
echo    VAID JI - Medical Research Assistant Setup
echo ================================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%
echo.

REM Create virtual environment
set ENV_NAME=medical_assistant_env
echo Creating virtual environment: %ENV_NAME%
python -m venv %ENV_NAME%
echo Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call %ENV_NAME%\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo Pip upgraded
echo.

REM Install dependencies
echo Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
echo Dependencies installed
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo .env file created
    echo.
    echo WARNING: Please edit .env and add your API keys:
    echo    - OPENROUTER_API_KEY (get from https://openrouter.ai/)
    echo    - GROQ_API_KEY (get from https://console.groq.com/)
    echo.
) else (
    echo .env file already exists
    echo.
)

REM Create necessary directories
echo Creating necessary directories...
if not exist chroma_db mkdir chroma_db
if not exist .sentence_transformers_cache mkdir .sentence_transformers_cache
echo Directories created
echo.

REM Run configuration validation
echo Validating configuration...
python config.py
echo.

echo ================================================
echo               Setup Complete! 
echo ================================================
echo.
echo Next steps:
echo 1. Activate the virtual environment:
echo    %ENV_NAME%\Scripts\activate.bat
echo.
echo 2. Make sure you've added API keys to .env file
echo.
echo 3. Run the application:
echo    streamlit run app.py
echo.
echo For more information, see README.md and QUICKSTART.md
echo.
echo Happy researching!
echo.
pause
