"""
api/main.py
KudiWise AI — FastAPI application.

Endpoints:
  POST /review       → Task A: review simulation
  POST /recommend    → Task B: personalized recommendation
  POST /chat         → Multi-turn conversational agent
  GET  /health       → Health check
  GET  /graph/viz    → LangGraph topology (text)
"""
from __future__ import annotations
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.models import (
    AgentState,
    ReviewRequest, ReviewResponse,
    RecommendRequest, RecommendResponse,
    ChatRequest, ChatResponse,
)
from agent.graph import kudiwise_graph

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("kudiwise.api")
settings = get_settings()


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("KudiWise AI starting up — env: %s", settings.app_env)
    # Warm up vectorstore on startup so first request isn't slow
    try:
        from core.vectorstore import get_vectorstore
        vs = get_vectorstore()
        logger.info("ChromaDB vectorstore ready")
    except Exception as exc:
        logger.warning("ChromaDB not available yet: %s", exc)
    yield
    logger.info("KudiWise AI shutting down")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="KudiWise AI",
    description=(
        "Behavioral Survival Recommendation & Review Simulation Agent "
        "for Nigerian Students. DSN × BCT Hackathon 3.0."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Error handler ───────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. The agent encountered a problem."},
    )


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "service": "KudiWise AI",
        "env": settings.app_env,
        "model": settings.gemini_llm_model,
    }


# ─── Graph visualisation ─────────────────────────────────────────────────────

@app.get("/graph/viz", tags=["System"])
async def graph_viz():
    """Return the LangGraph node topology as text — useful for the solution paper."""
    try:
        diagram = kudiwise_graph.get_graph().draw_ascii()
        return {"topology": diagram}
    except Exception:
        return {
            "topology": (
                "START → classify_intent → "
                "[review: fetch_few_shot → run_task_a] | "
                "[recommend: retrieve_items_node → run_task_b] | "
                "[clarify: ask_clarification] | "
                "[general: general_chat] → END"
            )
        }


# ─── POST /review — Task A ───────────────────────────────────────────────────

@app.post("/review", response_model=ReviewResponse, tags=["Task A"])
async def review(req: ReviewRequest):
    """
    Task A — User Modeling.
    Simulates how a financially stressed Nigerian student would
    rate and review the given item.
    """
    logger.info("/review | item=%s price=₦%s", req.item_name, req.price_ngn)

    initial_state = AgentState(
        persona=req.persona,
        user_input=f"Review {req.item_name} priced at ₦{req.price_ngn}",
        intent="review",
        item_name=req.item_name,
        price_ngn=req.price_ngn,
    )

    try:
        result: AgentState = await kudiwise_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Graph error in /review: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if result.error:
        raise HTTPException(status_code=422, detail=result.error)

    if not result.review_output:
        raise HTTPException(status_code=500, detail="Agent produced no review output.")

    logger.info("/review done | rating=%s steps=%s", result.review_output.rating, result.steps_taken)
    return result.review_output


# ─── POST /recommend — Task B ─────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse, tags=["Task B"])
async def recommend(req: RecommendRequest):
    """
    Task B — Personalized Recommendation.
    Retrieves real items from Amazon / Yelp / Goodreads datasets
    and ranks them by survival value for the given student persona.
    """
    logger.info("/recommend | need=%s budget=₦%s", req.need, req.persona.weekly_budget_ngn)

    initial_state = AgentState(
        persona=req.persona,
        user_input=req.need,
        intent="recommend",
        need=req.need,
    )

    try:
        result: AgentState = await kudiwise_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Graph error in /recommend: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if result.error:
        raise HTTPException(status_code=422, detail=result.error)

    if not result.recommend_output:
        raise HTTPException(status_code=500, detail="Agent produced no recommendation output.")

    logger.info(
        "/recommend done | recs=%s cold_start=%s steps=%s",
        len(result.recommend_output.recommendations),
        result.recommend_output.cold_start,
        result.steps_taken,
    )
    return result.recommend_output


# ─── POST /chat — Multi-turn ──────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Conversational multi-turn agent.
    Automatically routes to review, recommend, or general chat
    based on the user's message. Maintains history client-side.
    """
    session_id = req.session_id or str(uuid.uuid4())
    logger.info("/chat | session=%s message=%s", session_id, req.message[:60])

    initial_state = AgentState(
        persona=req.persona,
        user_input=req.message,
        history=req.history,
    )

    try:
        result: AgentState = await kudiwise_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Graph error in /chat: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # Build unified chat response regardless of which path was taken
    reply = result.chat_reply
    items = None
    review_out = None

    if result.recommend_output:
        reply = reply or f"Here are your top picks based on your ₦{req.persona.weekly_budget_ngn:,} budget:"
        items = result.recommend_output.recommendations
    elif result.review_output:
        reply = reply or f"{result.review_output.rating}★ — {result.review_output.review}"
        review_out = result.review_output

    if not reply:
        reply = "I'm here to help! Tell me what you need or what item you want reviewed."

    return ChatResponse(
        reply=reply,
        intent=result.intent or "general",
        items=items,
        review=review_out,
        session_id=session_id,
    )
