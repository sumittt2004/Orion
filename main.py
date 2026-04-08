"""
main.py — Entry point for the Autonomous Research Agent.

Usage:
    python main.py "What are the latest breakthroughs in quantum computing?"

Or run interactively (no argument):
    python main.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule

from config import cfg
from graph import compile_graph
from state import AgentState

console = Console()


# ── Pretty printing helpers ────────────────────────────────────────────────

NODE_LABELS = {
    "planner":         "[bold purple]Planner[/]",
    "searcher":        "[bold red]Web Search[/]",
    "memory_retrieve": "[bold cyan]Memory Recall[/]",
    "memory_store":    "[bold cyan]Memory Store[/]",
    "synthesizer":     "[bold purple]Synthesizer[/]",
    "critic":          "[bold purple]Critic[/]",
}


def _print_step(node_name: str, state: dict) -> None:
    label = NODE_LABELS.get(node_name, node_name)
    console.print(f"  → {label}", end=" ")

    if node_name == "planner":
        qs = state.get("sub_questions", [])
        console.print(f"generated {len(qs)} sub-questions (iteration {state.get('iteration', '?')})")

    elif node_name == "searcher":
        n = len(state.get("search_results", []))
        console.print(f"fetched {n} total results so far")

    elif node_name == "memory_retrieve":
        n = len(state.get("memory_chunks", []))
        console.print(f"recalled {n} memory chunks")

    elif node_name == "memory_store":
        console.print("persisted new findings to FAISS")

    elif node_name == "synthesizer":
        words = len(state.get("draft_report", "").split())
        console.print(f"draft ready ({words} words)")

    elif node_name == "critic":
        fb = state.get("critic_feedback", {})
        score = fb.get("score", 0)
        done  = fb.get("is_complete", False)
        status = "[green]✓ accepted[/]" if done else "[yellow]✗ needs more research[/]"
        console.print(f"score={score:.2f} {status}")

    else:
        console.print("")


def _save_report(query: str, report: str) -> Path:
    out_dir = Path(cfg.REPORT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    slug      = "".join(c if c.isalnum() else "_" for c in query[:50]).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"{slug}_{timestamp}.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n\n**Query:** {query}\n\n---\n\n{report}\n")

    return out_path


# ── Main run loop ──────────────────────────────────────────────────────────

def run(query: str) -> str:
    console.print(Rule("[bold]Autonomous Research Agent[/]"))
    console.print(Panel(f"[italic]{query}[/italic]", title="Query", border_style="blue"))
    console.print()

    app = compile_graph()

    initial_state: AgentState = {
        "query":          query,
        "sub_questions":  [],
        "search_results": [],
        "memory_chunks":  [],
        "draft_report":   "",
        "critic_feedback": {},        # type: ignore[typeddict-item]
        "iteration":      0,
        "is_complete":    False,
        "final_report":   "",
        "messages":       [],
    }

    console.print("[dim]Running graph…[/dim]")

    final_state = initial_state
    for node_name, state in app.stream(initial_state, stream_mode="updates"):
        # state is a dict of {node_name: partial_state}
        if isinstance(state, dict) and node_name in state:
            partial = state[node_name]
            # Merge into final_state for display purposes
            final_state = {**final_state, **partial}
            _print_step(node_name, final_state)

    report = final_state.get("final_report") or final_state.get("draft_report", "")
    if not report:
        console.print("[red]Agent produced no report. Check API keys and try again.[/red]")
        return ""

    console.print()
    console.print(Rule("[bold green]Final Report[/]"))
    console.print(Markdown(report))

    out_path = _save_report(query, report)
    console.print()
    console.print(f"[dim]Report saved to:[/dim] [bold]{out_path}[/bold]")

    return report


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        console.print("[bold]Autonomous Research Agent[/bold]")
        user_query = console.input("[blue]Enter your research query:[/blue] ").strip()
        if not user_query:
            console.print("[red]No query provided. Exiting.[/red]")
            sys.exit(1)

    if not cfg.ANTHROPIC_API_KEY:
        console.print("[red]ANTHROPIC_API_KEY is not set. Add it to your .env file.[/red]")
        sys.exit(1)

    if not cfg.TAVILY_API_KEY:
        console.print("[red]TAVILY_API_KEY is not set. Add it to your .env file.[/red]")
        sys.exit(1)

    run(user_query)