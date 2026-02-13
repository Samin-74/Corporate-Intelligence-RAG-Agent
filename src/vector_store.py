"""
Vector Store Module
- Manages ChromaDB for local vector storage
- Handles embedding generation via sentence-transformers
- Supports adding documents and querying
"""

import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
)
from src.ingestion import DocumentChunk


class VectorStore:
    """ChromaDB-backed vector store with sentence-transformer embeddings."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ):
        self.persist_dir = persist_dir or str(CHROMA_DB_DIR)
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL_NAME

        # Initialize embedding function (runs on CPU)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name,
            device=EMBEDDING_DEVICE,
        )

        # Initialize ChromaDB persistent client
        self._client = chromadb.PersistentClient(path=self.persist_dir)

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return self._collection.count()

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        # Generate unique IDs with timestamp to prevent collisions
        import hashlib
        ids = []
        for c in chunks:
            # Create hash from content + metadata for uniqueness
            unique_str = f"{c.text}_{c.source_file}_{c.page_number}_{c.chunk_index}"
            chunk_hash = hashlib.md5(unique_str.encode()).hexdigest()[:8]
            ids.append(f"{c.source_file}_p{c.page_number}_c{c.chunk_index}_{chunk_hash}")

        documents = [c.text for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        # ChromaDB handles embedding automatically via the embedding function
        self._collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def query(
        self,
        query_text: str,
        n_results: int = 10,
    ) -> list[dict]:
        """
        Query the vector store for similar chunks.

        Args:
            query_text: The search query.
            n_results: Number of results to return.

        Returns:
            List of dicts with keys: 'text', 'metadata', 'distance'
        """
        # Handle empty queries
        if not query_text or not query_text.strip():
            return []

        # Handle empty collection
        if self.count == 0:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.count),
            include=["documents", "metadatas", "distances"],
        )

        # Flatten results (ChromaDB returns nested lists)
        output = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                })

        return output

    def clear(self) -> None:
        """Delete the collection and recreate it (fresh start)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_by_source(self, source_file: str) -> None:
        """Remove all chunks from a specific source file."""
        self._collection.delete(
            where={"source_file": source_file}
        )

    def list_sources(self) -> list[str]:
        """Get a list of all unique source files in the store."""
        if self.count == 0:
            return []
        # Get all metadata
        result = self._collection.get(include=["metadatas"])
        sources = set()
        for meta in result["metadatas"]:
            if "source_file" in meta:
                sources.add(meta["source_file"])
        return sorted(sources)
