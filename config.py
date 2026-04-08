# -*- coding: utf-8 -*-
"""
config.py - Central configuration for Orion Research Agent.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── API Keys ──────────────────────────────────────────────────────────
    GROQ_API_KEY:   str = os.getenv("GROQ_API_KEY",   "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # ── Model ─────────────────────────────────────────────────────────────
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    MAX_TOKENS: int = 4096

    # ── Planner ───────────────────────────────────────────────────────────
    MAX_SUB_QUESTIONS: int = 5

    # ── Tavily Search ─────────────────────────────────────────────────────
    SEARCH_RESULTS_PER_QUERY: int = 5
    SEARCH_MAX_CHARS: int = 2000

    # ── FAISS Memory ──────────────────────────────────────────────────────
    EMBEDDING_MODEL: str  = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./faiss_store")
    MEMORY_TOP_K: int     = 5

    # ── Critic / Loop ─────────────────────────────────────────────────────
    MAX_ITERATIONS: int    = 3
    QUALITY_THRESHOLD: float = 0.8

    # ── Output ────────────────────────────────────────────────────────────
    REPORT_OUTPUT_DIR: str = os.getenv("REPORT_OUTPUT_DIR", "./reports")


cfg = Config()