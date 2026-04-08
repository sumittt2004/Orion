# -*- coding: utf-8 -*-
"""
graph.py - Builds and compiles the LangGraph research agent.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from state import AgentState
from agent_nodes.planner     import planner_node
from agent_nodes.searcher    import searcher_node
from agent_nodes.memory      import memory_retrieve_node, memory_store_node
from agent_nodes.synthesizer import synthesizer_node
from agent_nodes.critic      import critic_node


def _should_continue(state: AgentState) -> str:
    if state.get("is_complete", False):
        return "end"
    return "planner"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner",         planner_node)
    graph.add_node("searcher",        searcher_node)
    graph.add_node("memory_retrieve", memory_retrieve_node)
    graph.add_node("memory_store",    memory_store_node)
    graph.add_node("synthesizer",     synthesizer_node)
    graph.add_node("critic",          critic_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "searcher")
    graph.add_edge("planner", "memory_retrieve")
    graph.add_edge("searcher", "memory_store")
    graph.add_edge("memory_retrieve", "synthesizer")
    graph.add_edge("memory_store",    "synthesizer")
    graph.add_edge("synthesizer", "critic")

    graph.add_conditional_edges(
        "critic",
        _should_continue,
        {"end": END, "planner": "planner"},
    )

    return graph


def compile_graph():
    return build_graph().compile()