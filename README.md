# 🧠 Edge-Optimized Corporate Intelligence RAG Agent

A **local, privacy-focused Financial Analyst chatbot** that runs entirely offline on consumer hardware. Query sensitive internal documents (PDFs) without any data leaving your machine — no cloud APIs, no costs, no privacy risks.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![VRAM](https://img.shields.io/badge/VRAM-<4GB-orange)
![Cost](https://img.shields.io/badge/Cost-$0-brightgreen)

---

## 🎯 Key Features

- **100% Local & Offline** — No API keys, no cloud services, no data leakage
- **GPU-Optimized** — Runs on an RTX 3050 Ti (4GB VRAM) via 4-bit quantization
- **Hybrid Search** — Vector search + keyword matching + cross-encoder reranking for maximum accuracy
- **Source Attribution** — Every answer shows exactly which document page was used
- **Real-Time Streaming** — Token-by-token generation for responsive UX
- **Financial Data Optimized** — Special handling for tables, numbers, and structured data

## 🏗️ Architecture

```
PDF Upload → Semantic Chunking → Vector Embeddings (ChromaDB)
                                        ↓
User Query → Vector Search (Top 25) ───┐
           → Keyword Search (Top 15) ──┼→ Merge & Deduplicate
                                       └→ Cross-Encoder Re-Rank (Top 8)
                                                  ↓
                                          Keyword Boost + Placeholder Penalty
                                                  ↓
                                          LLM Generation (Streamed)
                                                  ↓
                                          Answer + Source Citations
```

**Key Innovation:** Hybrid retrieval combines semantic understanding (vector search) with exact text matching (keyword search) to handle both natural language queries and precise financial data lookups. The cross-encoder reranking is boosted for chunks with exact keyword matches and penalized for placeholder symbols, ensuring actual data ranks higher than templates.

## 🛠️ Technology Stack

| Component        | Technology                          | Why                                              |
| ---------------- | ----------------------------------- | ------------------------------------------------ |
| **LLM**          | Phi-4 Mini (3.8B, Q4_K_M GGUF)     | Best reasoning at 3.8B params, ~2.3GB VRAM       |
| **Inference**    | llama-cpp-python (CUDA)             | GPU-accelerated GGUF inference                   |
| **Embeddings**   | all-MiniLM-L6-v2                    | Fast, 22M params, runs on CPU                    |
| **Vector DB**    | ChromaDB                            | Local file-based, no server needed               |
| **Re-Ranker**    | ms-marco-MiniLM-L-6-v2             | Cross-Encoder accuracy boost on CPU              |
| **PDF Parsing**  | PyMuPDF                             | Fastest Python PDF extraction                    |
| **Frontend**     | Streamlit                           | Rapid interactive UI                             |

## 📋 Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (tested on RTX 3050 Ti, 4GB VRAM)
- **CUDA Toolkit 12.x** installed
- **Visual Studio Build Tools** (for compiling llama-cpp-python on Windows)

## 🚀 Quick Start

### Option 1: One-Click Startup (Recommended)

**Windows:**
```batch
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

The startup script handles everything: environment check, model download, and app launch.

### Option 2: Manual Setup

### 1. Clone & Setup Environment

```bash
git clone <your-repo-url>
cd RAG
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install base dependencies
pip install -r requirements.txt

# Install llama-cpp-python with CUDA support (Windows)
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python

# Linux/Mac
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### 3. Run the Application

```bash
streamlit run app.py
```

The model will automatically download on first run (~2-3 minutes with progress indicators).

Open `http://localhost:8501` in your browser. Upload a PDF and start chatting!

## 📁 Project Structure

```
RAG/
├── app.py                  # Streamlit UI with auto-download & auto-config
├── start.bat / start.sh    # One-click startup scripts
├── requirements.txt        # Python dependencies
├── config.py               # Centralized configuration
├── .env.example            # Environment variable template
├── CLOUD_DEPLOY.md         # Cloud deployment guide
├── QUICK_DEPLOY.md         # Quick deployment guide
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # PDF loading & chunking (Docling + PyMuPDF)
│   ├── vector_store.py     # ChromaDB operations
│   ├── retriever.py        # Hybrid search (vector + keyword + reranking)
│   ├── llm.py              # LLM loading & generation
│   └── rag_pipeline.py     # Orchestrates the full pipeline
├── models/                 # GGUF model files (auto-downloaded, gitignored)
├── data/
│   └── chroma_db/          # Vector database (gitignored)
└── sample_docs/            # Sample PDFs for testing
```
|                         | Value          | Notes                                  |
| ----------------------- | ------------ | ---------------------------------------- |
| VRAM Usage              | ~3.0 GB      | Phi-4 Mini Q4_K_M + embeddings           |
| Time to First Token     | < 1 second   | After initial model warmup               |
| Generation Speed        | 20-40 tok/s  | GPU-accelerated                          |
| Hybrid Search Latency   | < 300ms      | Vector + keyword + cross-encoder         |
| Document Indexing       | ~2s / page   | Including embedding generation           |

## 🧪 Example Queries
 with hybrid search
- **Edge/Local AI Deployment** — Privacy-first, zero-cost inference
- **Quantized Model Optimization** — GGUF Q4, VRAM budgeting
- **Advanced Information Retrieval** — Keyword + Vector + Cross-Encoder hybrid search
- **Vector Databases** — ChromaDB embeddings & similarity search
- **GPU/CUDA Engineering** — Hardware-constrained ML deployment
- **Full-Stack Python** — Backend pipeline + Streamlit frontend
- **Production-Ready Code** — Error handling, logging, deployment scripts

## 🚢 Deployment

### Cloud Deployment ⭐ Recommended: Railway
See [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md) and [QUICK_DEPLOY.md](QUICK_DEPLOY.md) for hosting online:
- **Railway** (recommended - 10GB storage, auto-deploy, ~$5/month)
- **Fly.io** (50GB storage, developer-friendly)
- **AWS / GCP / Azure** (production-grade, unlimited storage)
- **Docker / Kubernetes** (containerized deployment)

⚠️ **Not Hugging Face Spaces** - 1GB storage limit is too small for 2.3GB model

**Quick Deploy to Railway:**
1. Create account at https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select `Samin-74/Corporate-Intelligence-RAG-Agent`
4. Railway auto-deploys! Model downloads automatically on first run (~5-7 min total)

See [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md) for complete instructions and cost comparisons.

**Conceptual Questions:**
- "What are the main risk factors?"
- "Summarize the company's strategy"
- "What segments does the company operate in?"

**Multi-hop Reasoning:**
- "How does revenue compare to the previous year?"
- "What factors contributed to the profit increase?"
| Time to First Token     | < 1 second   |
| Generation Speed        | 20-40 tok/s  |
| Re-ranking Latency      | < 200ms      |

## 🔑 Skills Demonstrated

- **Retrieval-Augmented Generation (RAG)** — End-to-end pipeline
- **Edge/Local AI Deployment** — Privacy-first, zero-cost inference
- **Quantized Model Optimization** — GGUF Q4, VRAM budgeting
- **Two-Stage Information Retrieval** — Bi-Encoder + Cross-Encoder
- **Vector Databases** — ChromaDB embeddings & similarity search
- **GPU/CUDA Engineering** — Hardware-constrained ML deployment
- **Full-Stack Python** — Backend pipeline + Streamlit frontend

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## ⚠️ Note on Dependencies

- PyMuPDF is licensed under AGPL-3.0 (free for personal/open-source use)
- All ML models used are MIT or Apache 2.0 licensed
- No paid APIs or services required
