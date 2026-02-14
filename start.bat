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
echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat

REM Start Streamlit (model auto-downloads on first run)
echo.
echo [NOTE] Model will download automatically on first run (~2-3 minutes)
echo.
echo [2/2] Starting Streamlit app...
echo Opening browser at http://localhost:8501
echo.
echo ========================================
echo   Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run app.py --server.headless true --server.maxUploadSize 200

pause
