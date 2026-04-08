# -*- coding: utf-8 -*-
"""
agent_nodes/searcher.py - Web search node powered by Tavily.
Client is initialized lazily so the API key is loaded from .env first.
"""
from __future__ import annotations

from typing import Any

from tavily import TavilyClient

from config import cfg
from state import AgentState, SearchResult


_tavily: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=cfg.TAVILY_API_KEY)
    return _tavily


def _truncate(text: str, max_chars: int = cfg.SEARCH_MAX_CHARS) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


def searcher_node(state: AgentState) -> dict[str, Any]:
    sub_questions = state["sub_questions"]
    seen_urls: set[str] = {r["url"] for r in state.get("search_results", [])}
    new_results: list[SearchResult] = []
    client = _get_client()

    for question in sub_questions:
        try:
            response = client.search(
                query=question,
                max_results=cfg.SEARCH_RESULTS_PER_QUERY,
                include_answer=False,
                search_depth="advanced",
            )
        except Exception as exc:
            print(f"[searcher] Tavily error for '{question}': {exc}")
            continue

        for item in response.get("results", []):
            url = item.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            new_results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=url,
                    content=_truncate(item.get("content", "")),
                    score=float(item.get("score", 0.0)),
                )
            )

    return {
        "search_results": new_results,
        "messages": [{"role": "searcher", "content": f"Found {len(new_results)} new results"}],
    }