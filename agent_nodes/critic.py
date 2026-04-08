# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any

from groq import Groq

from config import cfg
from state import AgentState, CriticFeedback

_client: Groq | None = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=cfg.GROQ_API_KEY)
    return _client

_SYSTEM = (
    "You are a rigorous research quality critic. Evaluate the draft report on "
    "coverage, accuracy, depth, coherence, and gaps. Return ONLY a valid JSON "
    "object with exactly these keys: score (float 0.0-1.0), is_complete (bool), "
    "missing_topics (list of strings), suggestions (string). "
    "No markdown, no explanation, just the JSON object. "
    "Set is_complete to true only if score >= {threshold}."
).format(threshold=cfg.QUALITY_THRESHOLD)


def critic_node(state: AgentState) -> dict[str, Any]:
    query     = state["query"]
    draft     = state.get("draft_report", "")
    iteration = state.get("iteration", 1)

    if iteration >= cfg.MAX_ITERATIONS:
        return {
            "critic_feedback": CriticFeedback(
                score=1.0, is_complete=True,
                missing_topics=[], suggestions="Max iterations reached."
            ),
            "is_complete":  True,
            "final_report": draft,
            "messages": [{"role": "critic", "content": "Max iterations reached."}],
        }

    response = _get_client().chat.completions.create(
        model=cfg.GROQ_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": f"Query:\n{query}\n\nDraft:\n{draft}"},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data     = json.loads(raw.strip())
    feedback = CriticFeedback(
        score=float(data.get("score", 0.5)),
        is_complete=bool(data.get("is_complete", False)),
        missing_topics=data.get("missing_topics", []),
        suggestions=data.get("suggestions", ""),
    )

    is_complete  = feedback["is_complete"]
    final_report = draft if is_complete else ""

    return {
        "critic_feedback": feedback,
        "is_complete":     is_complete,
        "final_report":    final_report,
        "messages": [{"role": "critic", "content": f"Score={feedback['score']:.2f}"}],
    }