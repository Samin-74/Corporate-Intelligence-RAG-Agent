"""
Model Downloader
Downloads the Phi-4 Mini GGUF model from HuggingFace.
Run once before starting the app.

Usage:  python download_model.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from config import MODELS_DIR, LLM_MODEL_REPO, LLM_MODEL_FILE, LLM_MODEL_PATH


def download_model():
    """Download the GGUF model from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    if LLM_MODEL_PATH.exists():
        size_gb = LLM_MODEL_PATH.stat().st_size / (1024 ** 3)
        print(f"✅ Model already exists: {LLM_MODEL_PATH}")
        print(f"   Size: {size_gb:.2f} GB")
        return

    print("=" * 60)
    print("  Edge RAG Agent — Model Downloader")
    print("=" * 60)
    print()
    print(f"  Repository : {LLM_MODEL_REPO}")
    print(f"  File       : {LLM_MODEL_FILE}")
    print(f"  Destination: {MODELS_DIR}")
    print(f"  Expected   : ~2.3 GB")
    print()
    print("  Downloading... (this may take a few minutes)")
    print()

    try:
        downloaded_path = hf_hub_download(
            repo_id=LLM_MODEL_REPO,
            filename=LLM_MODEL_FILE,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )

        # Verify download
        final_path = Path(downloaded_path)
        if final_path.exists():
            size_gb = final_path.stat().st_size / (1024 ** 3)
            print()
            print(f"  ✅ Download complete!")
            print(f"  📁 Saved to: {final_path}")
            print(f"  📏 Size: {size_gb:.2f} GB")
            print()
            print("  You can now run: streamlit run app.py")
        else:
            print(f"  ❌ Download failed — file not found at {final_path}")
            sys.exit(1)

    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        print()
        print("  Manual download instructions:")
        print(f"  1. Go to https://huggingface.co/{LLM_MODEL_REPO}")
        print(f"  2. Download '{LLM_MODEL_FILE}'")
        print(f"  3. Place it in: {MODELS_DIR}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
