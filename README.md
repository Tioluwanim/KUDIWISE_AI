# KudiWise AI

**Behavioral Survival Recommendation & Review Simulation Agent for Nigerian Students**

DSN × BCT LLM Agent Challenge — Hackathon 3.0

---

## What it does

| Task | Endpoint | Description |
|------|----------|-------------|
| A — User Modeling | `POST /review` | Simulates how a financially stressed Nigerian student rates and reviews a product |
| B — Recommendation | `POST /recommend` | Retrieves real items from Amazon/Yelp/Goodreads and ranks by survival value |
| Chat | `POST /chat` | Multi-turn conversational agent that auto-routes to review or recommend |

## Stack

- **LangGraph** — agent graph with intent routing
- **LangChain** — LCEL chains, prompt templates, memory
- **Gemini 1.5 Flash** — LLM generation (via `langchain-google-genai`)
- **Gemini text-embedding-004** — vector embeddings
- **ChromaDB** — vector store for dataset retrieval
- **FastAPI** — REST API
- **Docker** — containerised deployment

---

## Setup (without Docker)

```bash
# 1. Clone and enter
git clone <repo-url>
cd kudiwise

# 2. Create virtualenv
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY from https://aistudio.google.com

# 5. Index datasets (skip if using pre-built ChromaDB from repo)
python scripts/embed_and_index.py \
  --amazon  data/amazon.jsonl \
  --yelp    data/yelp.jsonl \
  --goodreads data/goodreads.jsonl \
  --limit 50000

# 6. Start the API
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
```

---

## Setup (with Docker)

```bash
cp .env.example .env   # add GOOGLE_API_KEY
docker-compose up --build
```

API available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

---

## API examples

### Task A — Review simulation
```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "persona": {
      "student_level": "300L",
      "field_of_study": "Computer Science",
      "weekly_budget_ngn": 8000,
      "urgency": "moderate",
      "location": "Lagos"
    },
    "item_name": "Wireless Earbuds",
    "price_ngn": 18000
  }'
```

### Task B — Recommendation
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "persona": {
      "student_level": "300L",
      "field_of_study": "Computer Science",
      "weekly_budget_ngn": 8000,
      "urgency": "survival",
      "location": "Abuja"
    },
    "need": "affordable study materials for exam"
  }'
```

### Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "persona": {"student_level": "200L", "weekly_budget_ngn": 5000, "urgency": "survival", "field_of_study": "Medicine", "location": "Ibadan"},
    "message": "I need something to eat that wont kill my budget",
    "history": []
  }'
```

---

## Run evaluation

```bash
# Start API first, then:
python scripts/evaluate.py --endpoint http://localhost:8000 --samples 10
# Results saved to data/eval_results.json
```

---

## Project structure

```
kudiwise/
├── api/
│   └── main.py          # FastAPI app — all endpoints
├── agent/
│   └── graph.py         # LangGraph agent — nodes + routing
├── core/
│   ├── config.py        # Settings from .env
│   ├── models.py        # All Pydantic schemas
│   ├── prompts.py       # All LangChain prompt templates
│   └── vectorstore.py   # ChromaDB singleton + retrieval
├── scripts/
│   ├── embed_and_index.py  # Dataset indexer
│   └── evaluate.py         # ROUGE, RMSE, Hit Rate evaluation
├── data/
│   └── chroma_db/          # Pre-built vector store (committed)
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## LangGraph agent topology

```
START
  └─► classify_intent
        ├─► [review]    fetch_few_shot ──► run_task_a ──► END
        ├─► [recommend] retrieve_items ──► run_task_b ──► END
        ├─► [clarify]   ask_clarification ────────────► END
        └─► [general]   general_chat ──────────────── ► END
```
