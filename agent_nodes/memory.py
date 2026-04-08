# -*- coding: utf-8 -*-
"""
agent_nodes/memory.py - FAISS long-term memory node.
Rewritten with explicit type guards to satisfy Pylance's strict checking
of the FAISS and SentenceTransformers stubs.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import cfg
from state import AgentState, MemoryChunk, SearchResult


# ── Constants ──────────────────────────────────────────────────────────────
_INDEX_FILE = Path(cfg.FAISS_INDEX_PATH) / "index.faiss"
_META_FILE  = Path(cfg.FAISS_INDEX_PATH) / "metadata.pkl"

# ── Module-level singletons ────────────────────────────────────────────────
_embedder: SentenceTransformer | None = None
_faiss_index: faiss.IndexFlatIP | None = None
_metadata: list[dict[str, str]] = []


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(cfg.EMBEDDING_MODEL)
    return _embedder


def _get_dim() -> int:
    dim = _get_embedder().get_sentence_embedding_dimension()
    if dim is None:
        raise ValueError("Could not determine embedding dimension from model.")
    return int(dim)


def _load_or_create_index() -> faiss.IndexFlatIP:
    global _faiss_index, _metadata
    if _faiss_index is not None:
        return _faiss_index

    if _INDEX_FILE.exists() and _META_FILE.exists():
        loaded = faiss.read_index(str(_INDEX_FILE))
        # Cast to IndexFlatIP so Pylance knows the concrete type
        _faiss_index = faiss.downcast_index(loaded)  # type: ignore[assignment]
        with open(_META_FILE, "rb") as f:
            _metadata = pickle.load(f)
    else:
        Path(cfg.FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        _faiss_index = faiss.IndexFlatIP(_get_dim())
        _metadata = []

    assert _faiss_index is not None
    return _faiss_index


def _save_index(index: faiss.IndexFlatIP) -> None:
    faiss.write_index(index, str(_INDEX_FILE))
    with open(_META_FILE, "wb") as f:
        pickle.dump(_metadata, f)


def _embed(texts: list[str]) -> np.ndarray:
    vecs = _get_embedder().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(vecs, dtype="float32")


# ── Node: retrieve ─────────────────────────────────────────────────────────

def memory_retrieve_node(state: AgentState) -> dict[str, Any]:
    queries = [state["query"]] + state.get("sub_questions", [])
    index   = _load_or_create_index()

    if index.ntotal == 0:
        return {
            "memory_chunks": [],
            "messages": [{"role": "memory", "content": "FAISS index empty."}],
        }

    query_vecs: np.ndarray = _embed(queries)
    agg_vec = query_vecs.mean(axis=0, keepdims=True).astype("float32")

    k = min(cfg.MEMORY_TOP_K, index.ntotal)
    distances = np.empty((1, k), dtype="float32")
    labels    = np.empty((1, k), dtype="int64")
    index.search(agg_vec, k, distances, labels)  # type: ignore[call-arg]

    chunks: list[MemoryChunk] = []
    seen: set[str] = set()
    for sim, idx in zip(distances[0].tolist(), labels[0].tolist()):
        if idx < 0 or idx >= len(_metadata):
            continue
        meta = _metadata[int(idx)]
        key  = meta["source_url"]
        if key in seen:
            continue
        seen.add(key)
        chunks.append(MemoryChunk(
            text=meta["text"],
            source_url=key,
            similarity=float(sim),
        ))

    return {
        "memory_chunks": chunks,
        "messages": [{"role": "memory", "content": f"Retrieved {len(chunks)} chunks"}],
    }


# ── Node: store ────────────────────────────────────────────────────────────

def memory_store_node(state: AgentState) -> dict[str, Any]:
    results: list[SearchResult] = state.get("search_results", [])
    if not results:
        return {"messages": [{"role": "memory", "content": "Nothing to store."}]}

    index = _load_or_create_index()
    existing_urls = {m["source_url"] for m in _metadata}
    new_entries   = [r for r in results if r["url"] not in existing_urls]

    if not new_entries:
        return {"messages": [{"role": "memory", "content": "All URLs already indexed."}]}

    texts = [f"{r['title']}\n\n{r['content']}" for r in new_entries]
    vecs  = _embed(texts)

    index.add(vecs)  # type: ignore[call-arg]
    _metadata.extend(
        {"text": t, "source_url": r["url"]}
        for t, r in zip(texts, new_entries)
    )
    _save_index(index)

    return {
        "messages": [{"role": "memory", "content": f"Stored {len(new_entries)} chunks"}],
    }