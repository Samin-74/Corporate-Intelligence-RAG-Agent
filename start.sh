#!/bin/bash
# Edge RAG Agent - Linux/Mac Startup Script

echo "========================================"
echo "  Edge RAG Agent - Startup"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run: python -m venv venv"
    echo "Then: source venv/bin/activate"
    echo "Then: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "[1/2] Activating virtual environment..."
source venv/bin/activate

# Start Streamlit (model auto-downloads on first run)
echo
echo "[NOTE] Model will download automatically on first run (~2-3 minutes)"
echo
echo "[2/2] Starting Streamlit app..."
echo "Opening browser at http://localhost:8501"
echo
echo "========================================"
echo "  Press Ctrl+C to stop the server"
echo "========================================"
echo

streamlit run app.py --server.headless true --server.maxUploadSize 200
