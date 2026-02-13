# Deployment Guide

## Quick Deploy (Recommended)

### Windows
```batch
start.bat
```

### Linux/Mac
```bash
chmod +x start.sh
./start.sh
```

The startup script will:
1. Check for virtual environment
2. Download the model if missing
3. Start Streamlit on http://localhost:8501

---

## Manual Deployment

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp-python with CUDA (Windows)
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Install llama-cpp-python with CUDA (Linux/Mac)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 2. Download Models

```bash
python download_model.py
```

This downloads `Phi-4-mini-instruct-Q4_K_M.gguf` (~2.3GB) to `models/`.

### 3. Run the Application

```bash
streamlit run app.py
```

Or with custom settings:
```bash
streamlit run app.py \
  --server.port 8501 \
  --server.headless true \
  --server.maxUploadSize 200
```

---

## Docker Deployment (Advanced)

### Build Image
```bash
docker build -t edge-rag-agent .
```

### Run Container
```bash
docker run -p 8501:8501 \
  --gpus all \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  edge-rag-agent
```

**Note:** Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

---

## Production Deployment

### Performance Tuning

1. **GPU Memory Optimization**
   - Adjust `N_GPU_LAYERS` in config.py based on available VRAM
   - Use Q4_K_M quantization for 4GB VRAM
   - Use Q8_0 quantization for 8GB+ VRAM

2. **Retrieval Performance**
   - Increase `RETRIEVAL_TOP_K` for better recall (slower)
   - Decrease `RERANK_TOP_N` for faster responses
   - Adjust `CHUNK_SIZE` based on document structure

3. **Concurrent Users**
   - Run multiple instances behind a load balancer
   - Each instance requires dedicated GPU memory
   - Share the ChromaDB directory (read-only for workers)

### Security

- **Network:** Bind to localhost only in production (`--server.address localhost`)
- **Authentication:** Use Streamlit's built-in authentication or reverse proxy (nginx + auth)
- **File Upload:** Limit upload size and file types (already configured: PDF only, 200MB max)
- **Data Privacy:** All processing is local, no data leaves the server

### Monitoring

- **Logs:** Check Streamlit logs for errors
- **GPU Usage:** `nvidia-smi -l 1` to monitor VRAM
- **Disk Space:** Monitor `data/chroma_db/` growth

---

## Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key settings:
- `LLM_MODEL_PATH`: Path to GGUF model file
- `N_GPU_LAYERS`: GPU offloading (-1 = all, 0 = CPU only)
- `RETRIEVAL_TOP_K`: Initial retrieval candidates
- `RERANK_TOP_N`: Final sources shown to LLM

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `N_GPU_LAYERS` in config.py
- Use a smaller quantization (Q4_K_M → Q4_0)
- Close other GPU applications

### Slow Generation
- Ensure GPU layers are enabled (`N_GPU_LAYERS=-1`)
- Check CUDA installation: `nvidia-smi`
- Verify llama-cpp-python was built with CUDA

### Model Not Found
- Run `python download_model.py`
- Check `models/` directory
- Verify model filename matches `LLM_MODEL_PATH`

### No Sources Found
- Check PDF uploaded successfully
- Try a different question
- Verify ChromaDB created: `data/chroma_db/` exists

---

## System Requirements

**Minimum:**
- Python 3.10+
- NVIDIA GPU with 4GB VRAM (RTX 3050 Ti or equivalent)
- CUDA Toolkit 12.x
- 8GB System RAM
- 10GB disk space (models + data)

**Recommended:**
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- 16GB System RAM
- SSD storage for faster vector DB access

---

## Updating

```bash
# Pull latest code
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart application
# Windows: start.bat
# Linux/Mac: ./start.sh
```

---

## Support

For issues, check:
1. This deployment guide
2. Main README.md
3. GitHub Issues
4. System logs in terminal output
