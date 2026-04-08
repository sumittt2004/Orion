# -*- coding: utf-8 -*-
from __future__ import annotations

import json
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

_SYSTEM = (
    "You are an expert research planner. Decompose the research query into "
    "3 to {max_q} precise, self-contained sub-questions each answerable via "
    "a single web search. If critic feedback is provided, add questions that "
    "fill the identified gaps. Return ONLY a valid JSON array of strings, "
    "no markdown, no explanation."
).format(max_q=cfg.MAX_SUB_QUESTIONS)


def planner_node(state: AgentState) -> dict[str, Any]:
    query     = state["query"]
    feedback  = state.get("critic_feedback")
    iteration = state.get("iteration", 0) + 1

    user_content = f"Research query: {query}"
    if feedback and not feedback.get("is_complete", True):
        missing     = ", ".join(feedback.get("missing_topics", []))
        suggestions = feedback.get("suggestions", "")
        user_content += (
            f"\n\nCritic feedback from iteration {iteration - 1}:\n"
            f"- Missing topics: {missing}\n"
            f"- Suggestions: {suggestions}\n\n"
            "Generate sub-questions that specifically address these gaps."
        )

    response = _get_client().chat.completions.create(
        model=cfg.GROQ_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_content},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    sub_questions: list[str] = json.loads(raw.strip())

    return {
        "sub_questions": sub_questions,
        "iteration":     iteration,
        "messages": [{"role": "planner", "content": sub_questions}],
    }