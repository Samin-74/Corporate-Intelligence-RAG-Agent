"""
Centralized configuration for the Edge RAG Agent.
All paths, model names, and hyperparameters in one place.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_docs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# LLM Configuration
# ──────────────────────────────────────────────
LLM_MODEL_REPO = "professorf/Phi-4-mini-instruct-gguf"
LLM_MODEL_FILE = "Phi-4-mini-instruct-q4_k_m.gguf"
LLM_MODEL_PATH = MODELS_DIR / LLM_MODEL_FILE

# Inference parameters
LLM_N_CTX = 4096          # Context window
LLM_N_GPU_LAYERS = -1     # -1 = offload ALL layers to GPU (0 = CPU only)
LLM_MAX_TOKENS = 1024     # Max tokens to generate (shorter = less repetition)
LLM_TEMPERATURE = 0.1     # Low temp for factual responses
LLM_TOP_P = 0.9
LLM_REPEAT_PENALTY = 1.15 # Moderate penalty to prevent repetition

# Backend selection: "llama_cpp" or "ollama"
# Use "ollama" if you have Ollama installed (handles CUDA automatically)
# Use "llama_cpp" for direct GGUF model loading
LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "llama_cpp")

# Ollama settings (only used if LLM_BACKEND == "ollama")
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi4-mini"  # Pull with: ollama pull phi4-mini

# ──────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"   # Keep on CPU, leave GPU for LLM

# ──────────────────────────────────────────────
# Cross-Encoder Re-Ranker
# ──────────────────────────────────────────────
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE = "cpu"    # Keep on CPU

# ──────────────────────────────────────────────
# Retrieval Parameters
# ──────────────────────────────────────────────
RETRIEVAL_TOP_K = 25       # Cast wider net for financial tables
RERANK_TOP_N = 8           # Keep top 8 after re-ranking for better coverage

# ──────────────────────────────────────────────
# Chunking Parameters
# ──────────────────────────────────────────────
CHUNK_SIZE = 1200          # Larger chunks to capture complete tables
CHUNK_OVERLAP = 150        # More overlap for table continuity

# ──────────────────────────────────────────────
# ChromaDB
# ──────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "documents"

# ──────────────────────────────────────────────
# Prompt Template (Phi-4 format)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a financial document analyst. Your job is to answer questions "
    "based ONLY on the provided source excerpts.\n\n"
    "Instructions:\n"
    "- Read ALL sources carefully before answering.\n"
    "- Use exact figures, names, and terms from the sources. Never guess.\n"
    "- Cite sources as (Page X).\n"
    "- Keep answers concise and factual. Do not repeat yourself.\n"
    "- If the answer is not in any source, say so briefly."
)

def format_prompt(context: str, question: str) -> str:
    """Format a prompt using the Phi-4 chat template."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\nSources:\n{context}\n\n"
        f"Question: {question}<|end|>\n"
        f"<|assistant|>\n"
    )
