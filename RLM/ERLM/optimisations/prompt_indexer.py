"""
O1: TF-IDF Prompt Indexer
=========================
Builds a TF-IDF index over a document at load time and exposes a
``search_context`` custom tool so the RLM can retrieve relevant chunks
in sub-linear time instead of scanning the full document linearly.
"""

from __future__ import annotations

from typing import Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "prompt_indexer requires scikit-learn. "
        "Install it with: pip install scikit-learn"
    ) from exc

import numpy as np


class PromptIndexer:
    """TF-IDF index over an arbitrary document string.

    At build time the document is split into overlapping fixed-size chunks
    and a TF-IDF matrix is computed.  At query time cosine similarity is
    used to rank chunks and the top-k results are returned as a formatted
    string that can be pasted directly into a model prompt.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.
    overlap:
        Number of characters shared between consecutive chunks.
    top_k:
        Default number of chunks to return from ``search``.
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        overlap: int = 200,
        top_k: int = 5,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

        self._chunks: list[str] = []
        self.chunk_offsets: list[tuple[int, int]] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None  # sparse matrix, shape (n_chunks, vocab)
        self._index_built: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, document: str) -> None:
        """Split *document* into overlapping chunks and fit a TF-IDF index.

        Parameters
        ----------
        document:
            The full document text to index.  If it is shorter than
            ``chunk_size`` it is treated as a single chunk.
        """
        if not document:
            self._chunks = []
            self.chunk_offsets = []
            self._index_built = False
            return

        self._chunks = []
        self.chunk_offsets = []

        step = max(1, self.chunk_size - self.overlap)
        start = 0
        doc_len = len(document)

        while start < doc_len:
            end = min(start + self.chunk_size, doc_len)
            self._chunks.append(document[start:end])
            self.chunk_offsets.append((start, end))
            if end == doc_len:
                break
            start += step

        # Fit TF-IDF on the chunks
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._chunks)
        self._index_built = True

    def search(self, query: str, top_k: int | None = None) -> str:
        """Return the top-k most relevant chunks for *query*.

        Parameters
        ----------
        query:
            Free-text search query.
        top_k:
            Override the instance-level default number of results.

        Returns
        -------
        str
            A formatted string with each matching chunk prefixed by its
            chunk number, character offsets, and similarity score.
            Returns an empty string if the index has not been built or
            the query is empty.
        """
        if not self._index_built or not self._chunks:
            return "[search_context] Index not built yet. Call build_index first."

        query = (query or "").strip()
        if not query:
            return "[search_context] Empty query — please provide search terms."

        k = top_k if top_k is not None else self.top_k
        k = min(k, len(self._chunks))

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]

        # Rank by score descending; use stable argsort for reproducibility
        ranked_indices = np.argsort(scores)[::-1][:k]

        parts: list[str] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            start, end = self.chunk_offsets[idx]
            score = float(scores[idx])
            chunk_text = self._chunks[idx]
            parts.append(
                f"[Chunk {rank} (chars {start}-{end}, score={score:.4f})]:\n{chunk_text}\n"
            )

        return "\n".join(parts) if parts else "[search_context] No relevant chunks found."

    def get_custom_tool(self) -> dict[str, Any]:
        """Return a custom-tool dict compatible with the RLM ``custom_tools`` API.

        The tool exposes ``search_context`` as a callable in the REPL and
        includes its docstring so the model understands when to use it.

        Returns
        -------
        dict
            Mapping of tool name → (callable, description).
        """
        return {
            "search_context": (
                self.search,
                (
                    "search_context(query, top_k=5) -> Returns top-k most relevant document "
                    "chunks for the query using TF-IDF cosine similarity. "
                    "Use this instead of linear scanning."
                ),
            )
        }
