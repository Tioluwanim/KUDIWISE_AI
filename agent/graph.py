"""
agent/graph.py
KudiWise AI LangGraph agent (Vertex AI Migration).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START

from core.config import get_settings
from core.models import AgentState, ReviewResponse, RecommendResponse, RecommendedItem
from core.prompts import task_a_prompt, task_b_prompt, intent_prompt, chat_prompt
from core.vectorstore import retrieve_items as vector_retrieve
logger = logging.getLogger(__name__)
settings = get_settings()


# ─── LLM singleton ───────────────────────────────────────────────────────────
def get_llm() -> ChatVertexAI:
    """Returns a Vertex AI LLM instance. Uses service account auth via environment."""
    return ChatVertexAI(
        model_name=settings.gemini_llm_model,
        temperature=0.7,
    )


# ─── JSON parsing helpers ────────────────────────────────────────────────────
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json(text: str) -> dict:
    return json.loads(_strip_code_fences(text))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_price(value: Any) -> str:
    try:
        if value is None or value == "":
            return "?"
        return f"₦{int(float(value)):,}"
    except Exception:
        return str(value) if value is not None else "?"


def _safe_float(value: Any, default: float = 0.5) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


# ─── Node: classify intent ───────────────────────────────────────────────────
def classify_intent(state: AgentState) -> AgentState:
    logger.info("[Node] classify_intent")
    llm = get_llm()
    chain = intent_prompt | llm
    response = chain.invoke({"message": state.user_input})

    try:
        parsed = _parse_json(response.content)
        state.intent = parsed.get("intent", "general")
        state.item_name = parsed.get("item_name")
        state.price_ngn = parsed.get("price_ngn")
        state.need = parsed.get("need")
    except Exception as exc:
        logger.warning("Intent parse failed: %s", exc)
        state.intent = "general"
        state.item_name = None
        state.price_ngn = None
        state.need = None

    state.steps_taken.append(f"classify_intent → {state.intent}")
    return state


# ─── Node: fetch few-shot examples ───────────────────────────────────────────
def fetch_few_shot(state: AgentState) -> AgentState:
    logger.info("[Node] fetch_few_shot")
    p = state.persona
    budget = _safe_int(getattr(p, "weekly_budget_ngn", 0))
    query = f"student review {state.item_name or 'product'} budget ₦{budget}"

    examples = vector_retrieve(
        query=query,
        budget_ngn=max(budget * 2, 0),
        k=5,
    )
    state.few_shot_examples = examples
    state.steps_taken.append(f"fetch_few_shot → {len(examples)} examples")
    return state


# ─── Node: run Task A — review simulation ────────────────────────────────────
def run_task_a(state: AgentState) -> AgentState:
    logger.info("[Node] run_task_a for: %s", state.item_name)
    llm = get_llm()
    chain = task_a_prompt | llm

    few_shot_examples = state.few_shot_examples or []
    few_shot_text = "\n".join(
        [
            f"- Item: {ex.get('item_name','Unknown')} | Rating: {ex.get('avg_rating','?')}★ | "
            f"Review excerpt: {str(ex.get('page_content',''))[:120]}..."
            for ex in few_shot_examples
        ]
    ) or "No examples available — use your best judgment."

    p = state.persona
    response = chain.invoke(
        {
            "student_level": p.student_level,
            "field_of_study": p.field_of_study,
            "location": p.location,
            "weekly_budget_ngn": p.weekly_budget_ngn,
            "urgency": p.urgency,
            "few_shot_examples": few_shot_text,
            "item_name": state.item_name or "Unknown item",
            "price_ngn": _safe_int(state.price_ngn, 0),
            "category": "General",
        }
    )

    try:
        parsed = _parse_json(response.content)
        rating = max(1, min(5, _safe_int(parsed.get("rating", 3))))
        state.review_output = ReviewResponse(
            rating=rating,
            review=str(parsed.get("review","")),
            value_score=_safe_float(parsed.get("value_score",0.5)),
            value_label=str(parsed.get("value_label","Okay value")),
            persona_summary=f"{p.student_level} | ₦{p.weekly_budget_ngn:,}/wk | {p.urgency}",
            reasoning=str(parsed.get("reasoning","")),
        )
    except Exception as exc:
        logger.error("Task A parse error: %s | raw: %s", exc, response.content[:200])
        state.error = f"Review generation failed: {exc}"

    state.steps_taken.append("run_task_a → review generated")
    return state


# ─── Helpers for Task B context ──────────────────────────────────────────────
def _build_retrieved_context(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "No items retrieved. Use general knowledge to help."
    lines: List[str] = []
    for i, item in enumerate(items, 1):
        domain = str(item.get("domain","unknown")).upper()
        name = item.get("item_name","Unknown")
        category = item.get("category","?")
        rating = item.get("avg_rating","?")
        price = item.get("price_ngn")
        price_text = _safe_price(price) if price is not None else ""
        line = f"{i}. [{domain}] {name} | Category: {category} | Avg rating: {rating}★"
        if price_text:
            line += f" | Price: {price_text}"
        lines.append(line)
    return "\n".join(lines)


# ─── Node: retrieve items for Task B ─────────────────────────────────────────
def retrieve_items_node(state: AgentState) -> AgentState:
    logger.info("[Node] retrieve_items")
    p = state.persona
    query = state.need or state.user_input
    budget = _safe_int(getattr(p,"weekly_budget_ngn",0))

    items = vector_retrieve(query=query, budget_ngn=budget, k=settings.retrieval_top_k)
    if len(items) < 3:
        logger.info("Cold-start triggered — using persona query")
        state.cold_start = True
        persona_query = f"{p.student_level} Nigerian student {p.field_of_study} survival budget ₦{budget} {p.urgency}"
        items = vector_retrieve(query=persona_query, budget_ngn=budget, min_rating=settings.cold_start_min_rating, k=settings.retrieval_top_k)
    else:
        state.cold_start = False

    state.retrieved_items = items
    state.steps_taken.append(f"retrieve_items → {len(items)} items (cold_start={state.cold_start})")
    return state


# ─── Node: run Task B — recommendation ───────────────────────────────────────
def run_task_b(state: AgentState) -> AgentState:
    logger.info("[Node] run_task_b")
    llm = get_llm()
    chain = task_b_prompt | llm
    p = state.persona
    retrieved_context = _build_retrieved_context(state.retrieved_items or [])

    response = chain.invoke({
        "student_level": p.student_level,
        "field_of_study": p.field_of_study,
        "location": p.location,
        "weekly_budget_ngn": p.weekly_budget_ngn,
        "urgency": p.urgency,
        "cold_start": state.cold_start,
        "retrieved_context": retrieved_context,
        "need": state.need or state.user_input,
    })

    try:
        parsed = _parse_json(response.content)
        recs: List[RecommendedItem] = []
        for item in parsed.get("recommendations", []):
            recs.append(RecommendedItem(
                rank=_safe_int(item.get("rank",0)),
                item_name=str(item.get("item_name","Unknown")),
                domain=str(item.get("domain","unknown")),
                category=item.get("category"),
                avg_rating=item.get("avg_rating"),
                reason=str(item.get("reason","")),
                survival_score=_safe_float(item.get("survival_score",0.5))
            ))
        state.recommend_output = RecommendResponse(
            recommendations=recs,
            cold_start=state.cold_start,
            query_summary=str(parsed.get("query_summary", state.need or "")),
            retrieved_count=len(state.retrieved_items or [])
        )
    except Exception as exc:
        logger.error("Task B parse error: %s | raw: %s", exc, response.content[:200])
        state.error = f"Recommendation failed: {exc}"

    state.steps_taken.append(f"run_task_b → {len(state.recommend_output.recommendations if state.recommend_output else [])} recs")
    return state


# ─── Node: ask clarification / general chat ──────────────────────────────────
def ask_clarification(state: AgentState) -> AgentState:
    logger.info("[Node] ask_clarification")
    p = state.persona
    state.chat_reply = f"I'm not quite sure what you need, {p.student_level} student. Are you looking to review a specific item, or do you want me to recommend something within your budget? Tell me more and I'll sort you out sharp sharp."
    state.intent = "clarify"
    state.steps_taken.append("ask_clarification")
    return state


def general_chat(state: AgentState) -> AgentState:
    logger.info("[Node] general_chat")
    llm = get_llm()
    p = state.persona
    chain = chat_prompt | llm

    history_messages = []
    for msg in state.history[-6:]:
        if msg.role == "user":
            history_messages.append(HumanMessage(content=msg.content))
        else:
            history_messages.append(AIMessage(content=msg.content))

    response = chain.invoke({
        "student_level": p.student_level,
        "field_of_study": p.field_of_study,
        "location": p.location,
        "weekly_budget_ngn": p.weekly_budget_ngn,
        "urgency": p.urgency,
        "history": history_messages,
        "message": state.user_input,
    })
    state.chat_reply = response.content
    state.steps_taken.append("general_chat → replied")
    return state


# ─── Build the LangGraph ─────────────────────────────────────────────────────
def route_after_intent(state: AgentState) -> str:
    intent = state.intent or "general"
    return {
        "review": "fetch_few_shot",
        "recommend": "retrieve_items_node",
        "clarify": "ask_clarification",
        "general": "general_chat"
    }.get(intent, "general_chat")


def build_graph() -> Any:
    builder = StateGraph(AgentState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("fetch_few_shot", fetch_few_shot)
    builder.add_node("run_task_a", run_task_a)
    builder.add_node("retrieve_items_node", retrieve_items_node)
    builder.add_node("run_task_b", run_task_b)
    builder.add_node("ask_clarification", ask_clarification)
    builder.add_node("general_chat", general_chat)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges("classify_intent", route_after_intent, {
        "fetch_few_shot": "fetch_few_shot",
        "retrieve_items_node": "retrieve_items_node",
        "ask_clarification": "ask_clarification",
        "general_chat": "general_chat",
    })
    builder.add_edge("fetch_few_shot", "run_task_a")
    builder.add_edge("run_task_a", END)
    builder.add_edge("retrieve_items_node", "run_task_b")
    builder.add_edge("run_task_b", END)
    builder.add_edge("ask_clarification", END)
    builder.add_edge("general_chat", END)

    return builder.compile()


kudiwise_graph: Any = build_graph()