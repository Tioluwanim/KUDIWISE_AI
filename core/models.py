"""
core/models.py
All shared Pydantic models using Enums for robust validation.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# ─── Enums (Replaces Literal) ─────────────────────────────────────────────────

class StudentLevel(str, Enum):
    L100 = "100"
    L200 = "200"
    L300 = "300"
    L400 = "400"
    L500 = "500"
    PG = "PG"

class UrgencyLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    SURVIVAL = "survival"

class ReviewDomain(str, Enum):
    AMAZON = "amazon"
    YELP = "yelp"
    GOODREADS = "goodreads"

class IntentType(str, Enum):
    REVIEW = "review"
    RECOMMEND = "recommend"
    CLARIFY = "clarify"
    GENERAL = "general"

# ─── Persona ─────────────────────────────────────────────────────────────────

class Persona(BaseModel):
    student_level: StudentLevel = StudentLevel.L300
    field_of_study: str = Field(default="Computer Science", max_length=80)
    weekly_budget_ngn: int = Field(default=8000, ge=500, le=100_000)
    urgency: UrgencyLevel = UrgencyLevel.MODERATE
    location: str = Field(default="Lagos", max_length=60)

# ─── Task A ──────────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    persona: Persona
    item_name: str = Field(min_length=2, max_length=200)
    price_ngn: int = Field(ge=0)
    category: Optional[str] = None
    domain: Optional[ReviewDomain] = ReviewDomain.AMAZON

class ReviewResponse(BaseModel):
    rating: int = Field(ge=1, le=5)
    review: str
    value_score: float = Field(ge=0.0, le=1.0)
    value_label: str
    persona_summary: str
    reasoning: str

# ─── Task B ──────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    persona: Persona
    need: str = Field(min_length=3, max_length=300)
    domains: Optional[List[ReviewDomain]] = None

from pydantic import BaseModel, Field, field_validator # Make sure to import field_validator

class RecommendedItem(BaseModel):
    rank: int
    item_name: str
    domain: str = "unknown"  # Acts as the fallback if the key is missing entirely
    category: Optional[str] = None
    avg_rating: Optional[float] = None
    reason: str
    survival_score: float

    @field_validator("domain", mode="before")
    @classmethod
    def handle_null_domain(cls, v: Any) -> str:
        """Forces explicit null/None values to a default string to prevent validation crashes."""
        if v is None:
            return "unknown"
        return str(v)

class RecommendResponse(BaseModel):
    recommendations: List[RecommendedItem]
    cold_start: bool
    query_summary: str
    retrieved_count: int

# ─── Chat / multi-turn ───────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    persona: Persona
    message: str = Field(min_length=1, max_length=1000)
    history: List[ChatMessage] = Field(default_factory=list)
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    intent: IntentType
    items: Optional[List[RecommendedItem]] = None
    review: Optional[ReviewResponse] = None
    session_id: Optional[str] = None

# ─── LangGraph agent state ───────────────────────────────────────────────────

class AgentState(BaseModel):
    persona: Persona
    user_input: str
    intent: Optional[str] = None
    item_name: Optional[str] = None
    price_ngn: Optional[int] = None
    need: Optional[str] = None
    retrieved_items: List[Dict[str, Any]] = Field(default_factory=list)
    cold_start: bool = False
    few_shot_examples: List[Dict[str, Any]] = Field(default_factory=list)
    review_output: Optional[ReviewResponse] = None
    recommend_output: Optional[RecommendResponse] = None
    chat_reply: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)
    error: Optional[str] = None
    steps_taken: List[str] = Field(default_factory=list)