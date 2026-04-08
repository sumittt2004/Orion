"""
state.py — Defines the AgentState that flows through the entire LangGraph.

Every node reads from and writes to this dict.  LangGraph merges partial
updates returned from each node, so nodes only need to return the keys they
changed.
"""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


# ── Supporting data classes ────────────────────────────────────────────────

class SearchResult(TypedDict):
    """A single result from Tavily."""
    title:   str
    url:     str
    content: str
    score:   float


class MemoryChunk(TypedDict):
    """A retrieved chunk from FAISS."""
    text:       str
    source_url: str
    similarity: float


class CriticFeedback(TypedDict):
    """Structured output from the Critic node."""
    score:           float          # 0.0 – 1.0
    is_complete:     bool
    missing_topics:  list[str]
    suggestions:     str


# ── Main state ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    query: str

    # Planner output
    sub_questions: list[str]

    # Search + memory retrieval
    search_results:  Annotated[list[SearchResult],  operator.add]
    memory_chunks:   Annotated[list[MemoryChunk],   operator.add]

    # Synthesiser output
    draft_report: str

    # Critic output
    critic_feedback: CriticFeedback

    # Loop control
    iteration:    int
    is_complete:  bool

    # Final output
    final_report: str

    # Scratchpad for internal node messages (not shown to user)
    messages: Annotated[list[dict[str, Any]], operator.add]