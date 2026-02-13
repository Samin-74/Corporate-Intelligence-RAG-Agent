"""
RAG Pipeline Orchestrator
- Ties together ingestion, vector store, retriever, and LLM
- Provides a single high-level API for the UI layer
"""

import time
from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import Generator

from config import CHUNK_SIZE, CHUNK_OVERLAP
from src.ingestion import process_pdf, DocumentChunk
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.llm import LLMGenerator


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0


class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrator.

    Usage:
        pipeline = RAGPipeline()
        pipeline.initialize()           # Load models
        pipeline.ingest_pdf("doc.pdf")  # Add documents
        response = pipeline.query("What is the revenue?")
    """

    def __init__(self):
        self._vector_store: VectorStore | None = None
        self._retriever: Retriever | None = None
        self._llm: LLMGenerator | None = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, skip_llm: bool = False) -> dict[str, float]:
        """
        Initialize all components. Call once at startup.

        Args:
            skip_llm: If True, skip loading the LLM (useful for testing ingestion).

        Returns:
            Dict of component load times in seconds.
        """
        timings = {}

        # 1. Vector Store (fast — just ChromaDB init)
        t0 = time.time()
        self._vector_store = VectorStore()
        timings["vector_store"] = time.time() - t0

        # 2. Retriever (loads cross-encoder model, ~2s on CPU)
        t0 = time.time()
        self._retriever = Retriever(vector_store=self._vector_store)
        timings["retriever"] = time.time() - t0

        # 3. LLM (loads GGUF into VRAM, ~5-15s)
        if not skip_llm:
            t0 = time.time()
            self._llm = LLMGenerator()
            llm_info = self._llm.load()
            timings["llm"] = time.time() - t0
            timings["llm_info"] = llm_info

        self._initialized = True
        return timings

    def ingest_pdf(self, pdf_path: str | Path, display_name: str | None = None) -> dict:
        """
        Process a PDF and add it to the vector store.

        Args:
            pdf_path: Path to the PDF file.
            display_name: Optional human-friendly filename to use in metadata
                          (useful when pdf_path is a temp file).

        Returns:
            Dict with ingestion stats.
        """
        if self._vector_store is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        t0 = time.time()

        # Process PDF into chunks
        chunks = process_pdf(
            pdf_path,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # Override source_file with friendly name if provided
        friendly_name = display_name or Path(pdf_path).name
        for chunk in chunks:
            chunk.source_file = friendly_name

        processing_time = time.time() - t0

        # Add to vector store
        t1 = time.time()
        num_added = self._vector_store.add_chunks(chunks)
        embedding_time = time.time() - t1

        return {
            "file": friendly_name,
            "num_chunks": num_added,
            "num_pages": len(set(c.page_number for c in chunks)),
            "processing_time_s": round(processing_time, 2),
            "embedding_time_s": round(embedding_time, 2),
            "total_time_s": round(processing_time + embedding_time, 2),
        }

    def query(self, question: str) -> RAGResponse:
        """
        Full RAG query: retrieve → re-rank → generate.

        Args:
            question: The user's question.

        Returns:
            RAGResponse with answer, sources, and timings.
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Validate and clean input
        question = question.strip()
        if not question:
            return RAGResponse(
                answer="Please provide a question.",
                sources=[],
            )

        total_start = time.time()

        # ── Retrieval + Re-ranking ──
        t0 = time.time()
        results = self._retriever.retrieve(question)
        retrieval_ms = (time.time() - t0) * 1000

        if not results:
            return RAGResponse(
                answer="No relevant documents found. Please upload a PDF first.",
                sources=[],
                retrieval_time_ms=retrieval_ms,
            )

        # Build context from top results — numbered, with truncation to fit context
        context_parts = []
        total_chars = 0
        max_context_chars = 6000  # Leave room for system prompt + question in 4K tokens

        for i, r in enumerate(results, 1):
            source = r['metadata']['source_file']
            page = r['metadata']['page_number']
            text = r['text'].strip()
            # Truncate individual chunks if too long
            if len(text) > 800:
                text = text[:800]
            entry = f"[Source {i}] {source}, Page {page}:\n{text}"
            if total_chars + len(entry) > max_context_chars:
                break
            context_parts.append(entry)
            total_chars += len(entry)

        context = "\n\n---\n\n".join(context_parts)

        # Build source list for UI
        sources = [
            {
                "file": r["metadata"]["source_file"],
                "page": r["metadata"]["page_number"],
                "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "score": round(r["relevance_score"], 4),
            }
            for r in results
        ]

        # ── Generation ──
        t0 = time.time()
        answer = self._llm.generate(context, question)
        generation_ms = (time.time() - t0) * 1000

        total_ms = (time.time() - total_start) * 1000

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieval_time_ms=round(retrieval_ms, 1),
            generation_time_ms=round(generation_ms, 1),
            total_time_ms=round(total_ms, 1),
        )

    def query_stream(self, question: str) -> tuple[Generator[str, None, None], list[dict], float]:
        """
        Streaming RAG query: retrieve → re-rank → stream generation.

        Returns:
            Tuple of (token_generator, sources, retrieval_time_ms)
        """
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Validate and clean input
        question = question.strip()
        if not question:
            def empty_gen():
                yield "Please provide a question."
            return empty_gen(), [], 0.0

        # ── Retrieval + Re-ranking ──
        t0 = time.time()
        results = self._retriever.retrieve(question)
        retrieval_ms = (time.time() - t0) * 1000

        if not results:
            def empty_gen():
                yield "No relevant documents found. Please upload a PDF first."
            return empty_gen(), [], retrieval_ms

        # Build context — numbered, with truncation
        context_parts = []
        total_chars = 0
        max_context_chars = 6000

        for i, r in enumerate(results, 1):
            source = r['metadata']['source_file']
            page = r['metadata']['page_number']
            text = r['text'].strip()
            if len(text) > 800:
                text = text[:800]
            entry = f"[Source {i}] {source}, Page {page}:\n{text}"
            if total_chars + len(entry) > max_context_chars:
                break
            context_parts.append(entry)
            total_chars += len(entry)

        context = "\n\n---\n\n".join(context_parts)

        sources = [
            {
                "file": r["metadata"]["source_file"],
                "page": r["metadata"]["page_number"],
                "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "score": round(r["relevance_score"], 4),
            }
            for r in results
        ]

        # ── Streaming Generation ──
        token_stream = self._llm.generate_stream(context, question)

        return token_stream, sources, round(retrieval_ms, 1)

    def get_document_count(self) -> int:
        """Number of chunks in the vector store."""
        if self._vector_store is None:
            return 0
        return self._vector_store.count

    def get_sources(self) -> list[str]:
        """List all ingested document names."""
        if self._vector_store is None:
            return []
        return self._vector_store.list_sources()

    def clear_documents(self) -> None:
        """Remove all documents from the vector store."""
        if self._vector_store is not None:
            self._vector_store.clear()

    def shutdown(self) -> None:
        """Release all resources."""
        if self._llm is not None:
            self._llm.unload()
        self._initialized = False
