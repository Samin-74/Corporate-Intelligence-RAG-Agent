---
title: Corporate Intelligence RAG Agent
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.29.0"
app_file: app.py
pinned: false
license: mit
hardware: cpu-basic
---

# 🧠 Corporate Intelligence RAG Agent

A **100% local, privacy-focused** document intelligence system for financial analysis. Upload PDFs, ask questions in natural language, and get instant answers with source citations.

## 🎯 Features

- **Complete Privacy** — No data leaves the server
- **Hybrid Search** — Vector + keyword + cross-encoder for maximum accuracy
- **Financial Data Optimized** — Special handling for tables and structured data
- **Source Attribution** — Every answer cites the exact page number
- **Real-time Streaming** — Token-by-token response generation

## 🚀 How to Use

1. **Click "Initialize System"** in the sidebar (first time only, loads models)
2. **Upload a PDF** — Financial statements, reports, contracts, etc.
3. **Ask questions** — "What is the gross profit?", "What are the main risks?", etc.
4. **Get instant answers** with source citations

## 📊 Example Questions

**Financial Documents:**
- What is the total revenue for 2025?
- What are the main risk factors?
- What is the company's gross profit margin?
- Summarize the cash flow statement

**General Documents:**
- What are the key findings?
- List all recommendations
- What methodology was used?

## 🛠️ Technology

- **LLM:** Phi-4 Mini (3.8B params, Q4_K_M quantization)
- **Embeddings:** all-MiniLM-L6-v2
- **Vector DB:** ChromaDB
- **Retrieval:** Hybrid search (vector + keyword + cross-encoder)
- **Frontend:** Streamlit

## 📖 Documentation

- [GitHub Repository](https://github.com/Samin-74/Corporate-Intelligence-RAG-Agent)
- [Cloud Deployment Guide](https://github.com/Samin-74/Corporate-Intelligence-RAG-Agent/blob/main/CLOUD_DEPLOY.md)
- [Local Setup Guide](https://github.com/Samin-74/Corporate-Intelligence-RAG-Agent/blob/main/README.md)

## ⚠️ Note

This Space runs in **CPU mode** for free tier compatibility. For GPU-accelerated inference (20-40 tok/s instead of 1-2 tok/s), upgrade to GPU hardware or deploy locally.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built with ❤️ for financial analysis and document intelligence**
