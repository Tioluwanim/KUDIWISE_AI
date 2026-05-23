"""
agent/graph.py
KudiWise AI LangGraph agent (Vertex AI Migration).
"""
from __future__ import annotations
import json
import logging

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START

from core.config import get_settings
from core.models import (
    AgentState, ReviewResponse, RecommendResponse, RecommendedItem
)
from core.prompts import (
    task_a_prompt, task_b_prompt, intent_prompt, chat_prompt
)
from core.vectorstore import retrieve_items as vector_retrieve

logger = logging.getLogger(__name__)
settings = get_settings()


# ─── LLM singleton ───────────────────────────────────────────────────────────

def get_llm() -> ChatVertexAI:
    """Returns a Vertex AI LLM instance. Uses GCP IAM auth via environment."""
    return ChatVertexAI(
        model_name=settings.gemini_llm_model,
        temperature=0.7,
    )


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    text = text.strip()

    # Handle markdown code blocks if present
    if text.startswith("```"):
        # Split by triple backticks
        parts = text.split("```")
        # parts[0] is usually empty, parts[1] is the content block
        if len(parts) > 1:
            content = parts[1].strip()
            # Strip language tags like 'json' or 'python'
            if content.startswith("json"):
                content = content[4:].strip()
            elif content.startswith("python"):
                content = content[6:].strip()
            text = content

    return json.loads(text)


# ─── Node: classify intent ───────────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """Use Vertex AI to classify user message into one of 4 intents."""
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
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("Intent parse failed: %s", exc)
        state.intent = "general"

    state.steps_taken.append(f"classify_intent → {state.intent}")
    return state


# ─── Node: fetch few-shot examples for Task A ────────────────────────────────

def fetch_few_shot(state: AgentState) -> AgentState:
    """Pull 5 real reviews from ChromaDB."""
    logger.info("[Node] fetch_few_shot")
    budget = state.persona.weekly_budget_ngn
    query = f"student review {state.item_name or 'product'} budget ₦{budget}"

    examples = vector_retrieve(
        query=query,
        budget_ngn=int(budget * 2),
        k=5,
    )

    state.few_shot_examples = examples
    state.steps_taken.append(f"fetch_few_shot → {len(examples)} examples")
    return state


# ─── Node: run Task A — review simulation ────────────────────────────────────

def run_task_a(state: AgentState) -> AgentState:
    """Generate a realistic review using few-shot examples + persona."""
    logger.info("[Node] run_task_a for: %s", state.item_name)
    llm = get_llm()
    chain = task_a_prompt | llm

    few_shot_text = "\n".join([
        f"- Item: {ex['item_name']} | Rating: {ex.get('avg_rating','?')}★ | "
        f"Review excerpt: {ex['page_content'][:120]}..."
        for ex in state.few_shot_examples
    ]) or "No examples available — use your best judgment."

    p = state.persona
    response = chain.invoke({
        "student_level": p.student_level,
        "field_of_study": p.field_of_study,
        "location": p.location,
        "weekly_budget_ngn": p.weekly_budget_ngn,
        "urgency": p.urgency,
        "few_shot_examples": few_shot_text,
        "item_name": state.item_name or "Unknown item",
        "price_ngn": state.price_ngn or 0,
        "category": "General",
    })

    try:
        parsed = _parse_json(response.content)
        rating = max(1, min(5, int(parsed.get("rating", 3))))
        state.review_output = ReviewResponse(
            rating=rating,
            review=parsed.get("review", ""),
            value_score=float(parsed.get("value_score", 0.5)),
            value_label=parsed.get("value_label", "Okay value"),
            persona_summary=f"{p.student_level} | ₦{p.weekly_budget_ngn:,}/wk | {p.urgency}",
            reasoning=parsed.get("reasoning", ""),
        )
    except Exception as exc:
        logger.error("Task A parse error: %s | raw: %s", exc, response.content[:200])
        state.error = f"Review generation failed: {exc}"

    state.steps_taken.append("run_task_a → review generated")
    return state


# ─── Node: retrieve items for Task B ─────────────────────────────────────────

def retrieve_items_node(state: AgentState) -> AgentState:
    """Embed the user's need and pull top-K items from ChromaDB."""
    logger.info("[Node] retrieve_items")
    p = state.persona
    query = state.need or state.user_input
    budget = p.weekly_budget_ngn

    items = vector_retrieve(
        query=query,
        budget_ngn=budget,
        k=settings.retrieval_top_k,
    )

    if len(items) < 3:
        logger.info("Cold-start triggered — using persona query")
        state.cold_start = True
        persona_query = (
            f"{p.student_level} Nigerian student {p.field_of_study} "
            f"survival budget ₦{budget} {p.urgency}"
        )
        items = vector_retrieve(
            query=persona_query,
            budget_ngn=budget,
            min_rating=settings.cold_start_min_rating,
            k=settings.retrieval_top_k,
        )
    else:
        state.cold_start = False

    state.retrieved_items = items
    state.steps_taken.append(
        f"retrieve_items → {len(items)} items (cold_start={state.cold_start})"
    )
    return state


# ─── Node: run Task B — recommendation ───────────────────────────────────────

def run_task_b(state: AgentState) -> AgentState:
    """Rank retrieved items with Vertex AI using survival-value reasoning."""
    logger.info("[Node] run_task_b")
    llm = get_llm()
    chain = task_b_prompt | llm
    p = state.persona

    if state.retrieved_items:
        context_lines = []
        for i, item in enumerate(state.retrieved_items, 1):
            line = (
                f"{i}. [{item['domain'].upper()}] {item['item_name']} "
                f"| Category: {item.get('category','?')} "
                f"| Avg rating: {item.get('avg_rating','?')}★ "
                f"| Price: ₦{item.get('price_ngn','?'):,}" if item.get("price_ngn")
                else f"{i}. [{item['domain'].upper()}] {item['item_name']}"
            )
            context_lines.append(line)
        retrieved_context = "\n".join(context_lines)
    else:
        retrieved_context = "No items retrieved. Use general knowledge to help."

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
        recs = []
        for item in parsed.get("recommendations", []):
            recs.append(RecommendedItem(
                rank=item.get("rank", 0),
                item_name=item.get("item_name", "Unknown"),
                domain=item.get("domain", "amazon"),
                category=item.get("category"),
                avg_rating=item.get("avg_rating"),
                reason=item.get("reason", ""),
                survival_score=float(item.get("survival_score", 0.5)),
            ))
        state.recommend_output = RecommendResponse(
            recommendations=recs,
            cold_start=state.cold_start,
            query_summary=parsed.get("query_summary", state.need or ""),
            retrieved_count=len(state.retrieved_items),
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
    state.chat_reply = (
        f"I'm not quite sure what you need, {p.student_level} student. "
        "Are you looking to review a specific item, or do you want me to "
        "recommend something within your budget? Tell me more and I'll sort you out sharp sharp."
    )
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
    routes = {
        "review": "fetch_few_shot",
        "recommend": "retrieve_items_node",
        "clarify": "ask_clarification",
        "general": "general_chat",
    }
    return routes.get(intent, "general_chat")

def build_graph() -> StateGraph:
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

kudiwise_graph = build_graph()