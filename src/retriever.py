"""
Retrieval & Re-Ranking Module — Hybrid Search
- Stage 1a: Dense vector retrieval from ChromaDB (semantic)
- Stage 1b: Keyword matching (exact text search for tables/numbers)
- Stage 2: Cross-Encoder re-ranking on merged candidates
"""

from __future__ import annotations

import re
from sentence_transformers import CrossEncoder

from config import (
    RERANKER_MODEL_NAME,
    RERANKER_DEVICE,
    RETRIEVAL_TOP_K,
    RERANK_TOP_N,
)
from src.vector_store import VectorStore


class Retriever:
    """Hybrid retriever: dense search + keyword search + cross-encoder re-ranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        reranker_model: str | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
    ):
        self.vector_store = vector_store
        self.top_k = top_k or RETRIEVAL_TOP_K
        self.top_n = top_n or RERANK_TOP_N

        # Load Cross-Encoder re-ranker
        model_name = reranker_model or RERANKER_MODEL_NAME
        self._cross_encoder = CrossEncoder(
            model_name,
            device=RERANKER_DEVICE,
        )

    def _keyword_search(self, query: str, max_results: int = 15) -> list[dict]:
        """
        Stage 1b: Find chunks that literally contain the query terms.
        Critical for financial tables where embedding similarity is low
        but exact text match is high.
        """
        if self.vector_store.count == 0:
            return []

        # Extract meaningful keywords (skip stopwords)
        stopwords = {
            "what", "is", "the", "a", "an", "of", "for", "in", "on", "to",
            "and", "or", "how", "much", "does", "do", "are", "was", "were",
            "has", "have", "had", "be", "been", "this", "that", "it", "its",
            "with", "from", "by", "at", "as", "can", "could", "would",
            "should", "will", "shall", "may", "might", "about", "which",
            "who", "whom", "where", "when", "why", "not", "no", "but",
            "if", "then", "than", "so", "just", "only", "also", "very",
        }

        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) >= 2]

        if not keywords:
            return []

        # Full phrase without stopwords
        key_phrase = " ".join(keywords)

        # Get ALL documents and score by keyword match
        all_docs = self.vector_store._collection.get(
            include=["documents", "metadatas"]
        )

        scored = []
        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
            doc_lower = doc.lower()

            # Count keyword hits
            keyword_hits = sum(1 for kw in keywords if kw in doc_lower)

            # Bonus for exact phrase match
            phrase_bonus = 3 if key_phrase in doc_lower else 0

            # Bonus for keywords in first 200 chars (headings)
            header_bonus = sum(1 for kw in keywords if kw in doc_lower[:200])

            total_score = keyword_hits + phrase_bonus + header_bonus

            if total_score > 0:
                scored.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": 0.0,
                    "keyword_score": total_score,
                })

        scored.sort(key=lambda x: x["keyword_score"], reverse=True)
        return scored[:max_results]

    def retrieve(self, query: str) -> list[dict]:
        """
        Hybrid retrieval pipeline:
        1a. Dense vector search -> Top K semantic candidates
        1b. Keyword search -> Top K text-match candidates
        2.  Merge & deduplicate
        3.  Cross-Encoder re-ranking with keyword boost -> Top N final results
        """
        # Extract keywords for boosting later
        stopwords = {
            "what", "is", "the", "a", "an", "of", "for", "in", "on", "to",
            "and", "or", "how", "much", "does", "do", "are", "was", "were",
            "has", "have", "had", "be", "been", "this", "that", "it", "its",
            "with", "from", "by", "at", "as", "can", "could", "would",
            "should", "will", "shall", "may", "might", "about", "which",
        }
        query_words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in query_words if w not in stopwords and len(w) >= 2]
        key_phrase = " ".join(keywords)

        # Stage 1a: Dense Retrieval
        vector_candidates = self.vector_store.query(
            query_text=query,
            n_results=self.top_k,
        )

        # Stage 1b: Keyword Search
        keyword_candidates = self._keyword_search(query, max_results=15)

        # Merge & Deduplicate (preserving keyword scores)
        # Use page + text hash for dedup key (text[:100] fails when pages share headers)
        def _dedup_key(c: dict) -> str:
            meta = c.get("metadata", {})
            page = meta.get("page_number", meta.get("page", ""))
            chunk_idx = meta.get("chunk_index", "")
            return f"{page}:{chunk_idx}:{c['text'][:60]}"

        seen_keys: dict[str, int] = {}   # dedup_key -> index in merged
        merged = []

        for c in vector_candidates:
            key = _dedup_key(c)
            if key not in seen_keys:
                seen_keys[key] = len(merged)
                merged.append(c)

        for c in keyword_candidates:
            key = _dedup_key(c)
            if key not in seen_keys:
                seen_keys[key] = len(merged)
                merged.append(c)
            else:
                # Already in merged (from vector) — transfer keyword_score
                idx = seen_keys[key]
                if "keyword_score" in c:
                    merged[idx]["keyword_score"] = c["keyword_score"]

        if not merged:
            return []

        # Stage 2: Cross-Encoder Re-Ranking with keyword boost
        pairs = [(query, c["text"]) for c in merged]
        scores = self._cross_encoder.predict(pairs)

        for candidate, score in zip(merged, scores):
            base_score = float(score)

            # Boost chunks that contain the exact search phrase
            doc_lower = candidate["text"].lower()
            boost = 0.0

            if key_phrase and key_phrase in doc_lower:
                boost += 5.0  # Big boost for exact phrase match

            # Boost for individual keyword matches
            kw_hits = sum(1 for kw in keywords if kw in doc_lower)
            if keywords:
                boost += (kw_hits / len(keywords)) * 3.0  # Up to 3.0 boost

            # Extra boost if keywords appear with numbers nearby (financial data)
            if kw_hits > 0 and re.search(r'\d{3,}', candidate["text"]):
                boost += 2.0  # Likely a financial statement with actual figures

            # Penalize chunks with placeholder symbols (no real data)
            placeholder_count = len(re.findall(r'\[●\]', candidate["text"]))
            if placeholder_count > 0:
                boost -= min(placeholder_count * 2.0, 15.0)

            # Direct boost from keyword search score (if chunk came from keyword search)
            if "keyword_score" in candidate:
                boost += candidate["keyword_score"] * 0.5

            candidate["relevance_score"] = base_score + boost

        ranked = sorted(
            merged,
            key=lambda x: x["relevance_score"],
            reverse=True,
        )

        return ranked[: self.top_n]
