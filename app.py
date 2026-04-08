"""
app.py — Orion Research Agent · Streamlit Web UI
Deployed on Hugging Face Spaces.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import time
import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Orion · Research Agent",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit menu & footer */
#MainMenu, footer { visibility: hidden; }

/* Sidebar brand */
.orion-brand {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.orion-tagline {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0;
    margin-bottom: 1.5rem;
}

/* Step cards */
.step-card {
    background: #1e293b;
    border-left: 3px solid #7c3aed;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    color: #cbd5e1;
}
.step-card.done { border-left-color: #10b981; }
.step-card.active { border-left-color: #f59e0b; }

/* Report container */
.report-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem 2rem;
}

/* Score pill */
.score-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.score-high { background: #064e3b; color: #6ee7b7; }
.score-low  { background: #7c2d12; color: #fca5a5; }
</style>
""", unsafe_allow_html=True)


# ── API key injection from HF Secrets ─────────────────────────────────────
# On HF Spaces, secrets are env vars. Locally, they come from .env.
os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
os.environ.setdefault("TAVILY_API_KEY",    os.getenv("TAVILY_API_KEY",    ""))

from config import cfg                                # noqa: E402
from graph  import compile_graph                      # noqa: E402
from state  import AgentState                         # noqa: E402


# ── Session state defaults ─────────────────────────────────────────────────
def _init_session():
    defaults = {
        "history":      [],   # list of {query, report, steps, score}
        "running":      False,
        "current_steps": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="orion-brand">🔭 Orion</p>', unsafe_allow_html=True)
    st.markdown('<p class="orion-tagline">Autonomous Research Agent</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
1. 🧠 **Planner** — breaks your query into sub-questions  
2. 🌐 **Web Search** — Tavily fetches live sources  
3. 💾 **Memory** — FAISS recalls past research  
4. ✍️ **Synthesizer** — Claude writes the report  
5. 🔍 **Critic** — scores quality, loops if needed  
""")

    st.markdown("---")
    st.markdown("**Settings**")
    max_iter = st.slider("Max research iterations", 1, 5, cfg.MAX_ITERATIONS)
    threshold = st.slider("Quality threshold", 0.5, 1.0, cfg.QUALITY_THRESHOLD, 0.05)

    st.markdown("---")
    if st.session_state.history:
        st.markdown(f"**Past queries ({len(st.session_state.history)})**")
        for i, h in enumerate(reversed(st.session_state.history)):
            short = h["query"][:40] + "…" if len(h["query"]) > 40 else h["query"]
            if st.button(short, key=f"hist_{i}", use_container_width=True):
                st.session_state["view_report"] = h
                st.rerun()

    st.markdown("---")
    st.caption("Built with Claude · Tavily · LangGraph · FAISS")


# ── Main area ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.markdown("## What do you want to research?")

    query = st.text_area(
        label="Research query",
        placeholder="e.g. What are the latest breakthroughs in quantum computing in 2025?",
        height=100,
        label_visibility="collapsed",
    )

    run_btn = st.button(
        "🔭 Start Research",
        type="primary",
        disabled=st.session_state.running or not query.strip(),
        use_container_width=True,
    )

    # ── Live steps panel ───────────────────────────────────────────────────
    steps_container = st.container()
    report_container = st.container()


with col_right:
    st.markdown("## Agent status")
    status_box = st.empty()
    status_box.info("Ready — enter a query to begin.")


# ── View a past report (from sidebar click) ────────────────────────────────
if "view_report" in st.session_state:
    h = st.session_state.pop("view_report")
    with report_container:
        st.markdown("---")
        st.markdown(f"### 📄 Report: *{h['query']}*")
        st.markdown(h["report"])
    st.stop()


# ── Run the agent ──────────────────────────────────────────────────────────
NODE_META = {
    "planner":         ("🧠", "Planner",        "Breaking query into sub-questions…"),
    "searcher":        ("🌐", "Web Search",      "Fetching live sources via Tavily…"),
    "memory_retrieve": ("💾", "Memory Recall",   "Searching FAISS for past research…"),
    "memory_store":    ("💾", "Memory Store",    "Persisting new findings to FAISS…"),
    "synthesizer":     ("✍️", "Synthesizer",     "Writing the research report…"),
    "critic":          ("🔍", "Critic",          "Evaluating report quality…"),
}


def _render_steps(steps: list[dict], active_node: str | None = None):
    with steps_container:
        if not steps:
            return
        st.markdown("### Agent trace")
        for s in steps:
            css = "done" if s.get("done") else ("active" if s["node"] == active_node else "")
            icon, label, _ = NODE_META.get(s["node"], ("⚙️", s["node"], ""))
            detail = s.get("detail", "")
            st.markdown(
                f'<div class="step-card {css}">'
                f'{icon} <strong>{label}</strong>'
                f'{"  —  " + detail if detail else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )


if run_btn and query.strip():
    # Validate keys
    if not cfg.GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Add it to HF Spaces Secrets (Settings → Variables).")
        st.stop()
    if not cfg.TAVILY_API_KEY:
        st.error("TAVILY_API_KEY is not set. Add it to HF Spaces Secrets.")
        st.stop()

    # Override config from sidebar sliders
    cfg.MAX_ITERATIONS    = max_iter
    cfg.QUALITY_THRESHOLD = threshold

    st.session_state.running      = True
    st.session_state.current_steps = []
    steps = st.session_state.current_steps

    status_box.warning("⏳ Research in progress…")

    app = compile_graph()
    initial: AgentState = {
        "query":           query,
        "sub_questions":   [],
        "search_results":  [],
        "memory_chunks":   [],
        "draft_report":    "",
        "critic_feedback": {},          # type: ignore[typeddict-item]
        "iteration":       0,
        "is_complete":     False,
        "final_report":    "",
        "messages":        [],
    }

    final_state = initial.copy()

    try:
        for update in app.stream(initial, stream_mode="updates"):
            if not isinstance(update, dict) or not update:
                continue
            node_name = next(iter(update))
            node_data = update[node_name]
            if not isinstance(node_data, dict):
                continue
            final_state = {**final_state, **node_data}

            icon, label, desc = NODE_META.get(node_name, ("⚙️", node_name, ""))

            # Build detail string
            detail = ""
            if node_name == "planner":
                n = len(node_data.get("sub_questions", []))
                detail = f"{n} sub-questions · iteration {node_data.get('iteration', '?')}"
            elif node_name == "searcher":
                n = len(final_state.get("search_results", []))
                detail = f"{n} sources collected"
            elif node_name == "memory_retrieve":
                n = len(final_state.get("memory_chunks", []))
                detail = f"{n} memory chunks recalled"
            elif node_name == "memory_store":
                detail = "FAISS index updated"
            elif node_name == "synthesizer":
                w = len(final_state.get("draft_report", "").split())
                detail = f"{w} words"
            elif node_name == "critic":
                fb = final_state.get("critic_feedback", {})
                score = fb.get("score", 0)
                done  = fb.get("is_complete", False)
                detail = f"score={score:.2f} · {'✅ accepted' if done else '🔁 iterating'}"

            steps.append({"node": node_name, "detail": detail, "done": True})
            _render_steps(steps)

        # ── Display final report ───────────────────────────────────────────
        report = final_state.get("final_report") or final_state.get("draft_report", "")
        fb     = final_state.get("critic_feedback", {})
        score  = fb.get("score", 0)
        iters  = final_state.get("iteration", 1)

        status_box.success(f"✅ Done · score {score:.0%} · {iters} iteration(s)")

        with report_container:
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Quality score",  f"{score:.0%}")
            c2.metric("Iterations",     iters)
            c3.metric("Sources",        len(final_state.get("search_results", [])))

            st.markdown("### 📄 Research Report")
            st.markdown(report)

            st.download_button(
                "⬇️ Download report (.md)",
                data=report,
                file_name=f"orion_report_{int(time.time())}.md",
                mime="text/markdown",
            )

        # Save to history
        st.session_state.history.append({
            "query":  query,
            "report": report,
            "steps":  steps.copy(),
            "score":  score,
        })

    except Exception as exc:
        status_box.error(f"Agent error: {exc}")
        raise

    finally:
        st.session_state.running = False