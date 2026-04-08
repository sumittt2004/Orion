# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

from groq import Groq

from config import cfg
from state import AgentState

_client: Groq | None = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=cfg.GROQ_API_KEY)
    return _client

_SYSTEM = """\
You are an expert research analyst. Synthesise the provided search results \
and memory into a comprehensive research report with these sections:
# <Title>
## Executive Summary
## Key Findings
## Detailed Analysis
## Sources
## Gaps & Limitations
Cite sources inline as [Title](URL). Be objective. Do not invent facts. Aim for 1000-1500 words."""


def synthesizer_node(state: AgentState) -> dict[str, Any]:
    query          = state["query"]
    search_results = state.get("search_results", [])
    memory_chunks  = state.get("memory_chunks",  [])

    search_block = "\n\n".join(
        f"[{i+1}] {r['title']} ({r['url']})\n{r['content']}"
        for i, r in enumerate(search_results)
    ) or "No search results available."

    memory_block = "\n\n".join(
        f"[Memory {i+1}] (source={c['source_url']})\n{c['text']}"
        for i, c in enumerate(memory_chunks)
    ) or "No prior memory available."

    user_message = (
        f"## Research Query\n{query}\n\n"
        f"## Web Search Results\n{search_block}\n\n"
        f"## Retrieved Memory\n{memory_block}\n\n"
        "Please write the research report now."
    )

    response = _get_client().chat.completions.create(
        model=cfg.GROQ_MODEL,
        max_tokens=cfg.MAX_TOKENS,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_message},
        ],
    )

    return {
        "draft_report": (response.choices[0].message.content or "").strip(),
        "messages": [{"role": "synthesizer", "content": "Draft report written."}],
    }