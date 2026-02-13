@echo off
REM Edge RAG Agent - Windows Startup Script

echo ========================================
echo   Edge RAG Agent - Startup
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if model exists
if not exist "models\Phi-4-mini-instruct-Q4_K_M.gguf" (
    echo.
    echo [WARNING] Model not found!
    echo Downloading Phi-4 Mini model... (this may take a few minutes)
    echo.
    python download_model.py
    if errorlevel 1 (
        echo [ERROR] Model download failed!
        pause
        exit /b 1
    )
)

REM Start Streamlit
echo.
echo [2/3] Starting Streamlit app...
echo [3/3] Opening browser at http://localhost:8501
echo.
echo ========================================
echo   Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run app.py --server.headless true --server.maxUploadSize 200

pause
