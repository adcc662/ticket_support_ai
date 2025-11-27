Ticket Support AI
==================

FastAPI service for classifying helpdesk tickets using local machine-learning models with an optional OpenAI fallback/hybrid mode. It exposes simple endpoints to classify single or multiple tickets, train models with your own data, and inspect configuration status.

Features
--------
- Classify tickets into categories and priorities with confidence scores.
- Three methods: `local`, `openai`, or `hybrid` (default; uses OpenAI when local confidence is low).
- Batch classification endpoint for bulk requests.
- Train or retrain lightweight scikit-learn models with your labeled tickets.
- Configure OpenAI credentials at runtime and check configuration status.
- Containerized dev setup with Docker + uv; hot reload included.

Architecture
------------
- **FastAPI** app defined in `src/main.py` with routes in `src/router.py`.
- **Service layer** in `src/service.py` handles preprocessing, local ML (Tfidf + MultinomialNB), OpenAI calls, and hybrid logic.
- **Schemas** and enums in `src/schemas.py`.
- Models saved under `/app/models` inside the container for persistence across reloads.

Prerequisites
-------------
- Python 3.12+ and [uv](https://github.com/astral-sh/uv) for local runs, or Docker/Compose.
- Optional: OpenAI API key for `openai`/`hybrid` methods (`OPENAI_API_KEY`).

Local Setup (uv)
----------------
```bash
# Install dependencies (without installing the project as a package)
uv sync --no-install-project

# Run the API with hot reload
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
Docs available at `http://localhost:8000/docs`.

Docker / Compose
----------------
```bash
# Build and start
docker compose up --build

# Rebuild only the API image if needed
docker compose build api
```
The container mounts the repo into `/app` and keeps models under `/app/models`.

Environment Variables
---------------------
- `OPENAI_API_KEY` (optional): enables OpenAI and hybrid modes automatically.
- `ENVIRONMENT` (optional): defaults to `development`; used for runtime context.
- `PYTHONPATH=/app/src` is set in `compose.yml` for module resolution.

API Endpoints (high level)
--------------------------
- `POST /classify-ticket` — classify a single ticket.
- `POST /classify-batch` — classify multiple tickets in one call.
- `POST /train-model` — train/retrain local models with labeled data.
- `POST /configure-openai` — set or update OpenAI credentials and parameters.
- `GET /openai-status` — check whether OpenAI is configured and which model is active.
- `GET /statistics` — view available categories/priorities and model status.

Request/Response Examples
-------------------------
### Classify a ticket
```bash
curl -X POST http://localhost:8000/classify-ticket \
  -H "Content-Type: application/json" \
  -d '{
        "title": "VPN not connecting",
        "description": "Cannot reach corporate network over VPN since morning",
        "classification_method": "hybrid"
      }'
```
Response (example):
```json
{
  "id_ticket": "8c2f1a4b",
  "category": "Network",
  "priority": "High",
  "category_confidence": 0.91,
  "priority_confidence": 0.87,
  "method_used": "openai_hybrid",
  "detected_feeling": "frustrated",
  "urgency_level": "high",
  "extracted_entities": "vpn, network",
  "reasoning": "..."
}
```

### Batch classification
```bash
curl -X POST http://localhost:8000/classify-batch \
  -H "Content-Type: application/json" \
  -d '[
        {"title": "Password reset", "description": "Forgot my password", "classification_method": "local"},
        {"title": "Email sync failing", "description": "Outlook cannot sync inbox", "classification_method": "hybrid"}
      ]'
```

### Train local models
```bash
curl -X POST http://localhost:8000/train-model \
  -H "Content-Type: application/json" \
  -d '{
        "tickets": [
          {"title": "Cannot log in", "description": "Password invalid", "category": "Login", "priority": "Medium"},
          {"title": "Laptop overheating", "description": "Fan noisy and very hot", "category": "Hardware", "priority": "High"}
        ]
      }'
```
Models are persisted to `/app/models/category_model.pkl` and `/app/models/priority_model.pkl` inside the container.

### Configure OpenAI
```bash
curl -X POST http://localhost:8000/configure-openai \
  -H "Content-Type: application/json" \
  -d '{
        "api_key": "sk-...",
        "model": "gpt-4.1-mini",
        "temperature": 0.2
      }'
```
You can also set `OPENAI_API_KEY` in the environment before starting the service.

Data Shapes
-----------
- Categories: `["Hardware", "Software", "Network", "Login", "Billing", "General", "Email", "Security"]`
- Priorities: `["High", "Medium", "Low"]`
- `classification_method`: one of `local`, `openai`, `hybrid` (optional; defaults to hybrid).

Notes and Limitations
---------------------
- OpenAI calls require a valid API key; without it the service falls back to local models.
- Local models are lightweight and meant for quick starts; provide richer training data via `/train-model` for better results.
- Entities and sentiment are heuristic in local mode; OpenAI mode produces richer signals when available.
