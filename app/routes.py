import logging

from fastapi import APIRouter, HTTPException, status

from app.agent import RecommendationAgent
from app.models import ChatRequest, ChatResponse, HealthResponse


router = APIRouter()
logger = logging.getLogger(__name__)

# One agent instance is enough because the API is stateless:
# every POST /chat request includes the full conversation history.
# The agent must not store per-user memory between requests.
agent = RecommendationAgent()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness endpoint used by deployment platforms and evaluators."""
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Return the next assistant message for a stateless conversation.

    Request flow:
    1. FastAPI receives JSON.
    2. Pydantic validates it as ChatRequest.
    3. The agent creates a ChatResponse.
    4. FastAPI serializes the response back to JSON.
    """
    try:
        # The agent returns a ChatResponse that already conforms to schema.
        return agent.respond(request)
    except ValueError as exc:
        # Value errors indicate a client-side issue (invalid request content).
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Catch-all to avoid leaking internal errors to clients.
        logger.exception("Unhandled error in /chat")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc
