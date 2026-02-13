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
echo "[1/3] Activating virtual environment..."
source venv/bin/activate

# Check if model exists
if [ ! -f "models/Phi-4-mini-instruct-Q4_K_M.gguf" ]; then
    echo
    echo "[WARNING] Model not found!"
    echo "Downloading Phi-4 Mini model... (this may take a few minutes)"
    echo
    python download_model.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Model download failed!"
        exit 1
    fi
fi

# Start Streamlit
echo
echo "[2/3] Starting Streamlit app..."
echo "[3/3] Opening browser at http://localhost:8501"
echo
echo "========================================"
echo "  Press Ctrl+C to stop the server"
echo "========================================"
echo

streamlit run app.py --server.headless true --server.maxUploadSize 200
