# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python with CUDA support
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip install --no-cache-dir llama-cpp-python

# Copy application code
COPY . .

# Create directories (model auto-downloads on first run)
RUN mkdir -p models data/chroma_db

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.maxUploadSize=200", \
     "--server.enableXsrfProtection=false", \
     "--server.enableCORS=false"]
