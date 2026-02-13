"""
Streamlit UI — Modified for Hugging Face Spaces Deployment
Uses huggingface_hub to download model automatically
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Download model from Hugging Face Hub on first run
@st.cache_resource(show_spinner=False)
def download_model_if_needed():
    """Download Phi-4 model from HuggingFace if not present."""
    from huggingface_hub import hf_hub_download
    import config
    
    model_path = Path(config.LLM_MODEL_PATH)
    
    if model_path.exists():
        return str(model_path)
    
    # Download from HuggingFace
    try:
        st.info("🔄 Downloading Phi-4 model from HuggingFace (first run only, ~2.3GB)...")
        downloaded_path = hf_hub_download(
            repo_id="microsoft/Phi-4",
            filename="phi-4-mini-instruct-Q4_K_M.gguf",
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False
        )
        return downloaded_path
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.info("💡 Alternative: Upload your own GGUF model file to the Space")
        return None

# Download model before importing config (which checks the path)
model_path = download_model_if_needed()

if model_path:
    import config
    config.LLM_MODEL_PATH = model_path

# Now import the rest of the app
# (Copy all the code from app.py here, or use exec/import)
from app import *

# Note: The rest of this file should be identical to app.py
# For deployment, either:
# 1. Copy all code from app.py below this line, OR
# 2. Import and run: exec(open('app.py').read())
