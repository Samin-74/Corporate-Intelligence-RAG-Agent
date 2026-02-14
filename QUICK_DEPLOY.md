# Quick Deployment Guide 🚀

Your RAG agent is now on GitHub with **auto-download** and **auto-configuration**!

---

## ✅ Code is Live

**GitHub Repository:** https://github.com/Samin-74/Corporate-Intelligence-RAG-Agent

All deployment files are ready:
- ✅ `CLOUD_DEPLOY.md` — Comprehensive cloud deployment guide
- ✅ `Dockerfile` — Container image for Docker/Kubernetes
- ✅ `docker-compose.yml` — Easy Docker Compose setup
- ✅ **Auto-download model** on first run with progress indicators
- ✅ **Auto-detect GPU/CPU** and configure accordingly

---

## 🎯 Fastest Deployment Options

### Option 1: Railway ⭐ RECOMMENDED

**Best for:** Quick deployment, automatic model download, production use  
**Cost:** Free tier available (~$5/month for hobby projects)  
**Setup time:** 5 minutes  
**Storage:** 10GB persistent (perfect for 2.3GB model!)

#### Steps:
1. **Create Railway account:** https://railway.app
2. **Create new project:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Connect your GitHub account
   - Select `Samin-74/Corporate-Intelligence-RAG-Agent`
3. **Railway auto-detects** Streamlit and deploys
4. **First run takes 5-7 minutes:**
   - Railway build: ~2-3 minutes
   - Model auto-download: ~2-3 minutes (shows progress in app)
   - Vectorstore creation: ~1 minute
5. **Done!** Your app is live with automatic model download

**No manual configuration needed** - the app handles everything!

---

### Option 2: Hugging Face Spaces ❌ NOT RECOMMENDED

**Why Not:** HF Spaces has **1GB storage limit**, but our model is **2.3GB**. Auto-download will fail.

**Use Railway, Fly.io, or Docker instead!**

<details>
<summary>If HF Spaces increases storage limits in future...</summary>

**Best for:** Testing, demos, portfolio  
**Cost:** Free (with GPU limits)  
**Setup time:** 5 minutes

#### Steps:
1. **Create HF account:** https://huggingface.co/join
2. **Create new Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `corporate-rag-agent`
   - SDK: **Streamlit**
   - Hardware: **CPU basic** (free) or **ZeroGPU** (free with limits)
   - Visibility: Public or Private

3. **Clone your Space locally:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/corporate-rag-agent
   cd corporate-rag-agent
   ```

4. **Copy files from your GitHub repo:**
   ```bash
   # Copy all necessary files
   cp -r ../RAG/* .


#### Steps:
1. **Create HF account:** https://huggingface.co/join
2. **Create new Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `corporate-rag-agent`
   - SDK: **Streamlit**
   - Hardware: **CPU basic** (free) or **ZeroGPU** (free with limits)
   - Visibility: Public or Private

3. **Clone your Space locally:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/corporate-rag-agent
   cd corporate-rag-agent
   ```

4. **Copy files from your GitHub repo:**
   ```bash
   # Copy all necessary files
   cp -r ../RAG/* .
   
   # Use HF-specific README
   cp README_HF_SPACES.md README.md
   ```

5. **Modify for HF Spaces (CPU mode):**
   
   Edit `config.py`:
   ```python
   # Change GPU layers to 0 for CPU mode
   N_GPU_LAYERS = 0  # CPU only for free tier
   ```

6. **Push to HF:**
   ```bash
   git add .
   git commit -m "Deploy RAG agent to HF Spaces"
   git push
   ```

7. **Wait 2-3 minutes** for build to complete

8. **Access your app** at `https://huggingface.co/spaces/YOUR_USERNAME/corporate-rag-agent`

**Note:** CPU mode is slower (1-2 tok/s vs 20-40 tok/s on GPU). Upgrade to ZeroGPU or T4 for better performance.

</details>

---

### Option 3: Docker (Self-Hosted)

**Best for:** Full control, private deployment  
**Cost:** Your server costs  
**Setup time:** 10 minutes

#### Requirements:
- Docker installed
- NVIDIA GPU + NVIDIA Container Runtime (for GPU support)

#### Steps:

1. **Clone your repo:**
   ```bash
   git clone https://github.com/Samin-74/Corporate-Intelligence-RAG-Agent.git
   cd Corporate-Intelligence-RAG-Agent
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access at:** `http://localhost:8501`

   **Note:** Model downloads automatically on first run (~2-3 minutes)

**For CPU-only mode:**
```bash
# Remove GPU requirements from docker-compose.yml
# Set N_GPU_LAYERS=0 in config.py
docker-compose up --build
```

---

### Option 3: Railway (Easy Cloud Deploy)

**Best for:** Quick cloud hosting with minimal config  
**Cost:** ~$5-20/month  
**Setup time:** 5 minutes

#### Steps:

1. **Sign up:** https://railway.app
2. **New Project** → **Deploy from GitHub repo**
3. **Select:** `Samin-74/Corporate-Intelligence-RAG-Agent`
4. **Add environment variables:**
   ```
   N_GPU_LAYERS=0
   STREAMLIT_SERVER_PORT=8080
   ```
5. **Deploy** — Railway auto-detects Streamlit
6. **Access via Railway-provided URL**

**Note:** Railway doesn't have free GPU tier. Use CPU mode (slower).

---

### Option 4: Replicate (API Access)

**Best for:** API-first deployment, pay-per-use  
**Cost:** ~$0.0001-0.001/second of GPU time  
**Setup time:** 15 minutes

#### Steps:

1. **Install Replicate CLI:**
   ```bash
   pip install replicate
   replicate login
   ```

2. **Create `cog.yaml`** (see CLOUD_DEPLOY.md for template)

3. **Push to Replicate:**
   ```bash
   cog push r8.im/your-username/rag-agent
   ```

4. **Use via API:**
   ```python
   import replicate
   
   output = replicate.run(
       "your-username/rag-agent",
       input={"pdf_file": open("doc.pdf", "rb"), "question": "What is the revenue?"}
   )
   ```

---

## 📊 Platform Comparison

| Platform           | Free Tier | GPU Support | Difficulty | Best For              |
| ------------------ | --------- | ----------- | ---------- | --------------------- |
| **HF Spaces**      | ✅ Yes     | ✅ ZeroGPU   | ⭐⭐        | Demos, portfolios     |
| **Docker**         | N/A       | ✅ Yes       | ⭐⭐⭐      | Self-hosted, privacy  |
| **Railway**        | ❌ No      | ❌ No        | ⭐         | Quick cloud deploy    |
| **Replicate**      | ❌ No      | ✅ Yes       | ⭐⭐       | API-first, pay-per-use|
| **AWS/GCP**        | ❌ No      | ✅ Yes       | ⭐⭐⭐⭐    | Production, scale     |

---

## 🎬 Model Hosting

The model file (`Phi-4-mini-instruct-Q4_K_M.gguf`, 2.3GB) is too large for GitHub.

### Options:

#### A. Download on Startup (Recommended for HF Spaces)
Already implemented in `app_hf_spaces.py`:
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="microsoft/Phi-4",
    filename="phi-4-mini-instruct-Q4_K_M.gguf"
)
```

#### B. Upload to Your Own HF Model Repo
```bash
# Install HF CLI
pip install huggingface_hub[cli]

# Login
huggingface-cli login

# Upload model
huggingface-cli upload your-username/phi4-gguf models/Phi-4-mini-instruct-Q4_K_M.gguf
```

Then update `app_hf_spaces.py`:
```python
repo_id="your-username/phi4-gguf"
```

#### C. Cloud Storage (AWS S3 / GCP / Azure)
Upload to cloud storage and download on startup:
```python
import boto3
s3 = boto3.client('s3')
s3.download_file('my-bucket', 'phi4.gguf', 'models/phi4.gguf')
```

---

## 🔐 Making it Production-Ready

### 1. Add Authentication

**Streamlit Secrets:**
```toml
# .streamlit/secrets.toml
[passwords]
admin = "your_secure_password_here"
```

```python
# In app.py
import streamlit as st

def check_password():
    if "authenticated" not in st.session_state:
        password = st.text_input("Password:", type="password")
        if st.button("Login"):
            if password == st.secrets["passwords"]["admin"]:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.stop()

check_password()
```

### 2. Enable HTTPS

For HF Spaces: Automatic ✅  
For Docker/Railway: Use Cloudflare or nginx with Let's Encrypt

### 3. Monitor Usage

Add logging:
```python
import logging

logging.basicConfig(
    filename='rag_agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Log queries
logging.info(f"Query: {question} | User: {user_id}")
```

---

## 📞 Next Steps

1. **Choose a platform** from options above
2. **Follow the steps** for that platform
3. **Test your deployment** with sample documents
4. **Share the URL** with users

For detailed instructions, see:
- [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md) — Complete cloud deployment guide
- [DEPLOY.md](DEPLOY.md) — Local/production deployment
- [README.md](README.md) — Project overview

---

## 💡 Tips

- **Start with HF Spaces** — It's free and easiest to set up
- **Use CPU mode initially** — Test everything works before paying for GPU
- **Monitor costs** — Set up billing alerts on paid platforms
- **Version your models** — Tag different model versions in git
- **Backup your data** — Export ChromaDB periodically

Good luck with your deployment! 🎉
