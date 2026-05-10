# App Package

Backend API code for the SHL assessment recommender.

## Request Flow

1. `main.py` creates the FastAPI app.
2. `routes.py` receives HTTP requests.
3. `models.py` validates request and response data with Pydantic.
4. `agent.py` orchestrates clarify, recommend, refine, compare, and refuse.
5. `retriever.py` performs hybrid search with sentence-transformers + FAISS.
6. `catalog_loader.py` loads and normalizes catalog data.
7. `prompts.py` stores Gemini prompts for grounded responses.

## Important Contract

The `/chat` response schema must never break:

```json
{
  "reply": "response",
  "recommendations": [],
  "end_of_conversation": false
}
```

During clarification or refusal, `recommendations` must be an empty array.

When recommendations are returned, each item may contain only:

- `name`
- `url`
- `test_type`
