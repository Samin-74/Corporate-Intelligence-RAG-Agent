# Cloud Deployment Guide

This guide covers deploying the Edge RAG Agent to various cloud platforms, making it accessible online for users.

---

## 📋 Deployment Considerations

### Auto-Download & Auto-Configuration ✨
The app now **automatically downloads the model on first run** with progress indicators! It also:
- Auto-detects GPU/CPU and configures accordingly
- Initializes the system without manual button clicks
- Shows clear status messages during download
- Displays hardware info in the sidebar

**No manual setup needed** - just deploy and run!

### Model File Information
The LLM model file (`Phi-4-mini-instruct-q4_k_m.gguf`, ~2.3GB) downloads automatically from Hugging Face Hub (`professorf/Phi-4-mini-instruct-gguf`).

**Storage Requirements:**
- ✅ **Railway:** 10GB persistent storage (works great!)
- ✅ **Fly.io:** 50GB persistent volumes (perfect!)
- ✅ **AWS/GCP/Azure:** Unlimited (no issues)
- ❌ **Hugging Face Spaces:** 1GB limit (too small for model)

### Resource Requirements
- **Minimum:** 8GB RAM, 3GB storage, 4GB GPU VRAM (T4 / RTX 3050 Ti)
- **Recommended:** 16GB RAM, 5GB storage, 8GB GPU VRAM (A10 / RTX 3060+)
- **CPU-only mode:** Possible but slow (1-2 tok/s vs 20-40 tok/s on GPU)

---

## 🚀 Option 1: Hugging Face Spaces ⚠️ NOT RECOMMENDED

**Why Not:** Hugging Face Spaces has a **1GB storage limit**, but our model is **2.3GB**. Auto-download will fail.

**Alternative:** Use Railway, Fly.io, AWS, GCP, or Azure instead (they have adequate storage).

<details>
<summary>If HF Spaces increases storage limits in future...</summary>

### Setup Steps

1. **Create Space on Hugging Face**
   ```bash
   # Visit https://huggingface.co/spaces
   # Click "Create new Space"
   # Choose: Streamlit SDK, GPU hardware (ZeroGPU or T4)
   ```

2. **Modify Code for Spaces**

   Create `app_hf.py` (Spaces entry point):
   ```python
   import os
   import streamlit as st
   from huggingface_hub import hf_hub_download

   # Download model on first run
   MODEL_REPO = "microsoft/Phi-4-mini-instruct-GGUF"
   MODEL_FILE = "Phi-4-mini-instruct-Q4_K_M.gguf"
   
   @st.cache_resource
   def download_model():
       model_path = hf_hub_download(
           repo_id=MODEL_REPO,
           filename=MODEL_FILE,
           cache_dir="./models"
       )
       return model_path

   # Rest of your app.py code
   # Update config.LLM_MODEL_PATH to use downloaded model
   ```

3. **Create `requirements.txt` for Spaces**
   ```text
   llama-cpp-python==0.2.32
   sentence-transformers
   chromadb
   docling
   PyMuPDF>=1.23.0
   streamlit
   huggingface-hub
   ```

4. **Add `README.md` at root (Spaces metadata)**
   ```yaml
   ---
   title: Corporate Intelligence RAG Agent
   emoji: 🧠
   colorFrom: blue
   colorTo: purple
   sdk: streamlit
   sdk_version: "1.29.0"
   app_file: app_hf.py
   pinned: false
   hardware: t4-small
   ---
   
   # Corporate Intelligence RAG Agent
   100% local RAG system now running on cloud!
   ```

5. **Push to Hugging Face**
   ```bash
   # Install git-lfs for large files
   git lfs install
   
   # Add Hugging Face remote (get URL from your Space page)
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/rag-agent
   
   # Push
   git push hf main
   ```

### Cost-Free ZeroGPU Alternative
```python
# Use ZeroGPU decorator for free GPU access
from zero_gpu_decorator import zero_gpu

@zero_gpu(duration=60)  # 60s GPU time per call
def generate_answer(question, context):
    # Your LLM generation code
    pass
```

---

## 🌩️ Option 2: Replicate (Easy Model Hosting)

**Pros:** Automatic GPU scaling, pay-per-use, model caching  
**Cons:** Costs money ($0.0001-0.001 per second), requires Replicate account

### Setup Steps

1. **Create Replicate Model**
   ```bash
   # Install Replicate CLI
   pip install replicate
   
   # Login
   replicate login
   ```

2. **Create `cog.yaml`**
   ```yaml
   build:
     gpu: true
     python_version: "3.10"
     python_packages:
       - "streamlit==1.29.0"
       - "llama-cpp-python[cuda]"
       - "sentence-transformers"
       - "chromadb"
   
   predict: "predict.py:Predictor"
   ```

3. **Create `predict.py`**
   ```python
   from cog import BasePredictor, Input, Path
   import config
   from src.rag_pipeline import RAGPipeline
   
   class Predictor(BasePredictor):
       def setup(self):
           """Load model once at startup"""
           self.pipeline = RAGPipeline()
           self.pipeline.initialize()
       
       def predict(
           self,
           pdf_file: Path = Input(description="PDF document"),
           question: str = Input(description="Question to ask")
       ) -> str:
           # Ingest PDF
           self.pipeline.ingest_pdf(str(pdf_file))
           
           # Query
           answer, sources, _ = self.pipeline.query(question)
           return f"{answer}\n\nSources: {sources}"
   ```

4. **Deploy**
   ```bash
   # Build and push
   cog push r8.im/your-username/rag-agent
   ```

5. **Access via API**
   ```python
   import replicate
   
   output = replicate.run(
       "your-username/rag-agent",
       input={
           "pdf_file": open("document.pdf", "rb"),
           "question": "What is the gross profit?"
       }
   )
   ```

---

## ☁️ Option 3: AWS (EC2 + ECS)

**Pros:** Full control, enterprise-grade, scalable  
**Cons:** More complex setup, costs money, requires AWS knowledge

### Architecture
```
ALB (Load Balancer) → ECS Fargate (Streamlit) → EFS (Model Storage)
                              ↓
                       S3 (Vector DB Backup)
```

### Setup Steps

1. **Create EFS for Model Storage**
   ```bash
   aws efs create-file-system \
     --performance-mode generalPurpose \
     --tags Key=Name,Value=rag-models
   ```

2. **Upload Model to EFS**
   ```bash
   # Mount EFS locally
   sudo mount -t efs fs-xxxxx:/ /mnt/efs
   
   # Copy model
   cp models/Phi-4-mini-instruct-Q4_K_M.gguf /mnt/efs/models/
   ```

3. **Create Dockerfile**
   ```dockerfile
   FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
   
   RUN apt-get update && apt-get install -y python3.10 python3-pip
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
   
   COPY . .
   
   # Model will be mounted from EFS at /app/models
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"]
   ```

4. **Push to ECR (Elastic Container Registry)**
   ```bash
   # Build
   docker build -t rag-agent .
   
   # Tag
   docker tag rag-agent:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/rag-agent:latest
   
   # Push
   docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/rag-agent:latest
   ```

5. **Create ECS Task Definition**
   ```json
   {
     "family": "rag-agent",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "4096",
     "memory": "16384",
     "containerDefinitions": [{
       "name": "streamlit",
       "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/rag-agent:latest",
       "portMappings": [{"containerPort": 8501}],
       "mountPoints": [{
         "sourceVolume": "efs-models",
         "containerPath": "/app/models"
       }],
       "resourceRequirements": [{
         "type": "GPU",
         "value": "1"
       }]
     }],
     "volumes": [{
       "name": "efs-models",
       "efsVolumeConfiguration": {
         "fileSystemId": "fs-xxxxx"
       }
     }]
   }
   ```

6. **Deploy via ECS Service**
   ```bash
   aws ecs create-service \
     --cluster rag-cluster \
     --service-name rag-service \
     --task-definition rag-agent \
     --desired-count 1 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx]}"
   ```

### Cost Estimate (us-east-1)
- **g4dn.xlarge (4 vCPU, 16GB RAM, T4 GPU):** ~$0.50/hour = $360/month
- **EFS Storage (5GB model):** ~$1.50/month
- **Data Transfer:** ~$0.09/GB outbound

---

## 🐳 Option 4: Google Cloud Run (Serverless)

**Pros:** Pay-per-use, auto-scaling, easy deployment  
**Cons:** Cold starts, GPU support limited (use CPU mode or Cloud Run GPU preview)

### Setup Steps

1. **Modify for CPU-only Mode**
   ```python
   # In config.py
   N_GPU_LAYERS = 0  # CPU only for Cloud Run
   ```

2. **Create `Dockerfile`**
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   # Model auto-downloads on first run
   
   EXPOSE 8080
   
   CMD streamlit run app.py --server.port=8080 --server.headless=true
   ```

3. **Deploy to Cloud Run**
   ```bash
   # Build and push to GCR
   gcloud builds submit --tag gcr.io/PROJECT_ID/rag-agent
   
   # Deploy (with 8GB RAM, 4 vCPU for better CPU inference)
   gcloud run deploy rag-agent \
     --image gcr.io/PROJECT_ID/rag-agent \
     --platform managed \
     --memory 8Gi \
     --cpu 4 \
     --timeout 300 \
     --max-instances 3
   ```

### Cost Estimate
- **With GPU (Cloud Run GPU preview):** ~$0.10/minute of GPU time
- **CPU-only:** ~$0.00002/vCPU-second + $0.0000025/GiB-second
- **Example:** 8GB RAM, 4 vCPU, 10k requests/month ≈ $10-30/month

---

## 🔧 Option 5: Railway / Fly.io ✅ RECOMMENDED

**Why These?** Both platforms have adequate storage (10GB+ persistent volumes) for auto-downloading the 2.3GB model.

**Pros:** Simple deployment, affordable, quick setup, automatic model download works perfectly  
**Cons:** No free GPU tier (CPU-only mode works but slower)

### Railway Setup (Easiest!)

1. **Create account** at https://railway.app
2. **Connect GitHub repo**: Click "New Project" → "Deploy from GitHub repo" → Select `Corporate-Intelligence-RAG-Agent`
3. **Add environment variables** (optional):
   ```
   STREAMLIT_SERVER_PORT=8080
   ```
4. **Railway auto-detects** Streamlit and deploys automatically!
5. **Model downloads on first run** (~2-3 minutes) with progress indicators in the app
6. **Upgrade to GPU instance** if needed (~$20-50/month for faster responses)

**First deployment takes ~5-7 minutes** (2-3 min for Railway build, 2-3 min for model download, 1 min for vectorstore creation)

### Fly.io Setup

1. **Install Fly CLI**
   ```bash
   brew install flyctl  # Mac
   # or
   pwsh -Command "iwr https://fly.io/install.ps1 -useb | iex"  # Windows
   ```

2. **Create `fly.toml`**
   ```toml
   app = "rag-agent"
   
   [build]
     dockerfile = "Dockerfile"
   
   [[services]]
     internal_port = 8501
     protocol = "tcp"
   
     [[services.ports]]
       port = 80
       handlers = ["http"]
   
     [[services.ports]]
       port = 443
       handlers = ["tls", "http"]
   
   [env]
     STREAMLIT_SERVER_PORT = "8501"
   
   [[vm]]
     size = "shared-cpu-4x"
     memory = "8gb"
   ```

3. **Deploy**
   ```bash
   fly launch
   fly deploy
   ```

---

## 📦 Model Hosting Solutions

### Built-in Auto-Download (Already Implemented!) ✅
The app automatically downloads the model from Hugging Face Hub on first run:
```python
# Already implemented in app.py - ensure_model_downloaded()
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="microsoft/Phi-4-mini-instruct-GGUF",
    filename="Phi-4-mini-instruct-Q4_K_M.gguf",
    cache_dir="./models"
)
```

**Upload your own model:**
```bash
huggingface-cli login
huggingface-cli upload your-username/phi4-gguf models/Phi-4-mini-instruct-Q4_K_M.gguf
```

### Option B: AWS S3
```python
import boto3
s3 = boto3.client('s3')

# Download from S3 on startup
s3.download_file('my-models-bucket', 'phi4.gguf', 'models/phi4.gguf')
```

### Option C: Google Cloud Storage
```python
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('my-models-bucket')
blob = bucket.blob('phi4.gguf')
blob.download_to_filename('models/phi4.gguf')
```

### Option D: CDN (Cloudflare R2, DigitalOcean Spaces)
```python
import requests

# Download from public CDN URL
response = requests.get('https://cdn.example.com/models/phi4.gguf', stream=True)
with open('models/phi4.gguf', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

---

## 🔐 Production Checklist

### Security
- [ ] Add authentication (Streamlit secrets, nginx auth, OAuth)
- [ ] Use HTTPS (Let's Encrypt, CloudFlare, ALB with ACM)
- [ ] Limit file upload size (already configured: 200MB)
- [ ] Rate limiting (nginx, CloudFlare, API Gateway)
- [ ] Input sanitization for queries

### Performance
- [ ] Enable GPU acceleration (N_GPU_LAYERS=-1)
- [ ] Cache embeddings model (`@st.cache_resource`)
- [ ] Use persistent storage for ChromaDB (EFS, EBS, Cloud Storage)
- [ ] Set up CDN for static assets
- [ ] Monitor GPU usage (CloudWatch, Prometheus)

### Monitoring
- [ ] Log aggregation (CloudWatch, Datadog, LogDNA)
- [ ] Error tracking (Sentry)
- [ ] Uptime monitoring (UptimeRobot, Pingdom)
- [ ] Cost tracking (AWS Cost Explorer, GCP Billing)

### Scaling
- [ ] Auto-scaling policies
- [ ] Load balancer health checks
- [ ] Database backups (ChromaDB snapshots)
- [ ] Model versioning

---

## 💰 Cost Comparison (Monthly, 24/7 operation)

| Platform              | GPU Type | Cost/Month | Free Tier           |
| --------------------- | -------- | ---------- | ------------------- |
| **Hugging Face**      | ZeroGPU  | $0-9       | ✅ Limited hours     |
| **Replicate**         | A40      | ~$100      | ❌ Pay per second    |
| **AWS EC2 (g4dn)**    | T4       | ~$360      | ❌ No free GPU       |
| **GCP Cloud Run GPU** | T4       | ~$50-150   | ❌ Pay per use       |
| **Railway**           | N/A      | ~$20-50    | ❌ No GPU on free    |
| **Fly.io**            | N/A      | ~$30-60    | ❌ No GPU            |

**Recommendation:** Start with **Hugging Face Spaces** (free ZeroGPU) for testing, then move to **AWS/GCP** for production if you need 24/7 availability.

---

## 🎯 Quick Deploy Commands

### Hugging Face Spaces
```bash
git remote add hf https://huggingface.co/spaces/your-username/rag-agent
git push hf main
```

### AWS ECS
```bash
docker build -t rag-agent .
docker tag rag-agent:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/rag-agent:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/rag-agent:latest
aws ecs update-service --cluster rag --service rag-service --force-new-deployment
```

### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/rag-agent
gcloud run deploy rag-agent --image gcr.io/PROJECT_ID/rag-agent --platform managed
```

---

## 📞 Support

For deployment issues:
1. Check platform-specific documentation
2. Review logs in cloud console
3. Verify model file is accessible
4. Check GPU availability and quotas
5. Monitor memory/VRAM usage

Good luck with your deployment! 🚀
