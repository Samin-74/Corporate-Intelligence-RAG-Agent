"""
Streamlit UI — Edge-Optimized Corporate Intelligence RAG Agent

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Lazy import config
import config


# ──────────────────────────────────────────────
# Auto-Download Model with Progress
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def ensure_model_downloaded():
    """Download model if not present, with progress indicator."""
    model_path = Path(config.LLM_MODEL_PATH)
    
    if model_path.exists():
        return str(model_path), "Model already available"
    
    # Model not found - download it
    try:
        from huggingface_hub import hf_hub_download
        import requests
        
        st.info("🔄 First run detected - downloading AI model...")
        
        with st.status("Downloading Phi-4 Mini model (~2.3GB)...", expanded=True) as status:
            status.write("📡 Connecting to Hugging Face Hub...")
            
            # Download with progress
            downloaded_path = hf_hub_download(
                repo_id=config.LLM_MODEL_REPO,
                filename=config.LLM_MODEL_FILE,
                local_dir=str(model_path.parent),
                local_dir_use_symlinks=False
            )
            
            status.write("✅ Model downloaded successfully!")
            status.update(label="Model ready", state="complete", expanded=False)
            
        return downloaded_path, "Downloaded successfully"
        
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.info("""**Alternative:** Run locally and download manually:
        ```bash
        python download_model.py
        ```""")
        return None, f"Download failed: {e}"


# ──────────────────────────────────────────────
# Auto-Detect GPU
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def detect_gpu():
    """Detect if GPU is available and return appropriate settings."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, gpu_name, vram_gb
    except:
        pass
    
    return False, None, 0


# ──────────────────────────────────────────────
# Auto-Configure Based on Hardware
# ──────────────────────────────────────────────
def auto_configure_system():
    """Auto-detect hardware and configure appropriately."""
    has_gpu, gpu_name, vram_gb = detect_gpu()
    
    if has_gpu:
        # GPU detected - use it
        config.LLM_N_GPU_LAYERS = -1  # All layers on GPU
        return f"GPU detected: {gpu_name} ({vram_gb:.1f}GB VRAM)"
    else:
        # CPU only
        config.LLM_N_GPU_LAYERS = 0
        return "CPU mode (no GPU detected)"


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Edge RAG Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Smooth typing indicator */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .thinking-indicator {
        animation: pulse 1.5s ease-in-out infinite;
        color: #888;
        font-style: italic;
    }
    /* Source cards */
    .source-card {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #4CAF50;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 4px 4px 0;
    }
    /* Timing info */
    .timing-info {
        font-size: 0.75rem;
        color: #888;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
def init_session_state():
    defaults = {
        "pipeline": None,
        "messages": [],
        "initialized": False,
        "ingested_files": set(),
        "llm_info": {},
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ──────────────────────────────────────────────
# Auto-Initialize on First Load
# ──────────────────────────────────────────────
if not st.session_state.initialized and "auto_init_attempted" not in st.session_state:
    st.session_state.auto_init_attempted = True
    
    # Auto-download model if needed
    model_path, download_msg = ensure_model_downloaded()
    
    if model_path:
        # Auto-configure based on hardware
        hardware_msg = auto_configure_system()
        
        # Auto-initialize the system
        with st.status("Initializing AI system...", expanded=True) as status:
            try:
                status.write(f"🖥️ {hardware_msg}")
                status.write("📦 Loading models...")
                
                from src.rag_pipeline import RAGPipeline
                
                pipeline = RAGPipeline()
                timings = pipeline.initialize()
                
                st.session_state.pipeline = pipeline
                st.session_state.initialized = True
                st.session_state.llm_info = timings.get("llm_info", {})
                st.session_state.llm_info["hardware"] = hardware_msg
                st.session_state.ingested_files = set(pipeline.get_sources())
                
                status.write("✅ System ready!")
                status.update(label="Ready to use", state="complete", expanded=False)
                time.sleep(0.5)
                st.rerun()
                
            except Exception as e:
                status.update(label="Initialization failed", state="error")
                st.error(f"❌ Initialization error: {e}")
                st.info("Please try refreshing the page. If the issue persists, check the logs.")


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Edge RAG Agent")
    st.caption("100% Local • Zero Cost • Auto-Configured")
    st.divider()

    # ── System Status ──
    st.subheader("⚡ System Status")

    if not st.session_state.initialized:
        st.info("🔄 Initializing system automatically...")
        st.caption("This may take a moment on first run.")
    else:
        st.success("✅ System Online")
        pipeline: RAGPipeline = st.session_state.pipeline
        doc_count = pipeline.get_document_count()
        sources = pipeline.get_sources()

        llm_info = st.session_state.llm_info
        backend_label = llm_info.get("backend", "unknown")
        model_label = llm_info.get("model", "?")
        hardware_info = llm_info.get("hardware", "Unknown")

        model_size_str = ""
        if backend_label == "llama_cpp" and Path(config.LLM_MODEL_PATH).exists():
            size_mb = Path(config.LLM_MODEL_PATH).stat().st_size / (1024 * 1024)
            model_size_str = f" ({size_mb:.0f}MB)"

        st.caption(f"🤖 {backend_label} — {model_label}{model_size_str}")
        st.caption(f"🖥️ {hardware_info}")
        st.metric("Chunks in DB", doc_count)

        if sources:
            st.caption("📄 Loaded documents:")
            for src in sources:
                st.caption(f"  • {src}")

    st.divider()

    # ── Document Upload ──
    st.subheader("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.session_state.initialized:
        from src.rag_pipeline import RAGPipeline

        pipeline: RAGPipeline = st.session_state.pipeline
        new_files_ingested = False

        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.ingested_files:
                ingest_status = st.status(f"Processing {uploaded_file.name}...", expanded=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                try:
                    ingest_status.write("📄 Extracting text from PDF...")
                    stats = pipeline.ingest_pdf(tmp_path, display_name=uploaded_file.name)
                    st.session_state.ingested_files.add(uploaded_file.name)
                    new_files_ingested = True
                    ingest_status.write(
                        f"✅ {stats['num_pages']} pages, "
                        f"{stats['num_chunks']} chunks ({stats['total_time_s']}s)"
                    )
                    ingest_status.update(
                        label=f"✅ {uploaded_file.name}", state="complete", expanded=False
                    )
                except Exception as e:
                    ingest_status.update(label=f"❌ {uploaded_file.name}", state="error")
                    st.error(str(e))
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

        if new_files_ingested:
            st.rerun()
    elif uploaded_files and not st.session_state.initialized:
        st.warning("⚠️ Initialize the system first!")

    st.divider()

    # ── Controls ──
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗂️ Clear DB", use_container_width=True):
            if st.session_state.initialized:
                st.session_state.pipeline.clear_documents()
                st.session_state.ingested_files = set()
                st.rerun()

    st.divider()

    # ── Help & Info ──
    with st.expander("ℹ️ Help & Tips"):
        st.markdown("""
        **How to use:**
        1. Initialize the system (loads models)
        2. Upload one or more PDF files
        3. Ask questions in natural language

        **Best practices:**
        - Be specific in your questions
        - Ask about data that's likely in the document
        - Check the source citations for accuracy

        **Performance:**
        - First query: ~10-15s (model warmup)
        - Subsequent queries: ~3-8s
        - Larger documents = longer indexing
        """)

    st.caption("Built with Phi-4 Mini (Q4_K_M) • ChromaDB • Hybrid Search • Streamlit")


# ──────────────────────────────────────────────
# Main Chat Interface
# ──────────────────────────────────────────────
st.header("💬 Chat with your Documents")

if not st.session_state.initialized:
    st.info("👈 Click **Initialize System** in the sidebar to get started.")
    st.markdown("""
    ### Welcome to Edge RAG Agent 🧠

    A **100% local, privacy-first** document intelligence system that runs entirely on your machine.

    **Features:**
    - 🔒 Complete privacy — no data leaves your computer
    - ⚡ GPU-accelerated inference on consumer hardware
    - 🎯 Hybrid search (vector + keyword + cross-encoder)
    - 📄 Support for PDF documents with source attribution
    - 💰 Zero cost — no API fees

    **Quick Start:**
    1. Click "Initialize System" in the sidebar
    2. Upload a PDF document  
    3. Ask questions and get instant answers with source citations
    """)
    st.stop()

if not st.session_state.ingested_files:
    st.info("📄 Upload a PDF in the sidebar to begin querying.")
    st.markdown("""
    ### Example Questions You Can Ask:
    
    **Financial Documents:**
    - What is the total revenue for 2025?
    - What are the main risk factors mentioned?
    - Summarize the cash flow statement
    - What is the company's gross profit margin?
    
    **General Documents:**
    - What are the key findings?
    - Summarize the main conclusions
    - What methodology was used?
    - List all recommendations
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} chunks used)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"**{i}. {src['file']}** — Page {src['page']} "
                            f"(relevance: {src['score']:.4f})"
                        )
                        st.caption(src["text"][:300])
                        if i < len(sources):
                            st.divider()

            if "timings" in message:
                t = message["timings"]
                st.caption(
                    f"⏱️ Retrieval: {t['retrieval_ms']}ms • "
                    f"Generation: {t['generation_ms']}ms • "
                    f"Total: {t['total_ms']}ms"
                )

# Chat input with placeholder
placeholder_text = "Ask a question about your documents..." if st.session_state.ingested_files else "Upload a document first to start chatting..."
if question := st.chat_input(placeholder_text, disabled=not st.session_state.ingested_files):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        pipeline: RAGPipeline = st.session_state.pipeline

        if pipeline.get_document_count() == 0:
            answer = "Please upload a PDF document first before asking questions."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            try:
                # Show search phase
                with st.status("Searching documents...", expanded=True) as search_status:
                    search_status.write("🔍 Searching vector database + keyword index...")
                    token_stream, sources, retrieval_ms = pipeline.query_stream(question)
                    search_status.write(f"✅ Found {len(sources)} relevant sources ({retrieval_ms}ms)")
                    search_status.update(label=f"Found {len(sources)} sources", state="complete", expanded=False)

                # Stream the response with visible generation indicator
                gen_start = time.time()
                full_response = ""
                first_token = True

                # Show "Generating..." until the first token arrives
                gen_indicator = st.status("Generating response...", expanded=True)
                gen_indicator.write("🧠 Model is thinking — this may take a moment...")
                response_placeholder = st.empty()

                for token in token_stream:
                    if first_token:
                        # First token arrived — close the indicator
                        gen_indicator.update(
                            label="Generating response...",
                            state="running",
                            expanded=False,
                        )
                        first_token = False
                    full_response += token
                    response_placeholder.markdown(full_response + " ▌")

                # Finalise
                gen_indicator.update(
                    label="Response complete",
                    state="complete",
                    expanded=False,
                )
                response_placeholder.markdown(full_response)
                generation_ms = round((time.time() - gen_start) * 1000, 1)
                total_ms = round(retrieval_ms + generation_ms, 1)

                # Show sources
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)} chunks used)", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f"**{i}. {src['file']}** — Page {src['page']} "
                                f"(relevance: {src['score']:.4f})"
                            )
                            st.caption(src["text"][:300])
                            if i < len(sources):
                                st.divider()

                st.caption(
                    f"⏱️ Retrieval: {retrieval_ms}ms • "
                    f"Generation: {generation_ms}ms • "
                    f"Total: {total_ms}ms"
                )

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                    "timings": {
                        "retrieval_ms": retrieval_ms,
                        "generation_ms": generation_ms,
                        "total_ms": total_ms,
                    },
                })
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
