from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


# Pydantic models define the API contract.
# They protect the schema from accidental changes and help FastAPI generate docs.
class Message(BaseModel):
    """One message in the stateless conversation history."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, description="Message text")


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="Full conversation history. The server stores no conversation state.",
    )


class Recommendation(BaseModel):
    """One SHL catalog recommendation returned by the agent."""

    # Keep this model intentionally small.
    # The assignment evaluator expects only these fields in recommendations.
    name: str
    url: HttpUrl
    test_type: str = Field(
        ...,
        description="Catalog test type code, such as K, P, A, S, B, C, or D.",
    )


class ChatResponse(BaseModel):
    """Response body for POST /chat.

    This schema should remain stable because automated evaluators depend on it.
    """

    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False

    @model_validator(mode="after")
    def validate_recommendations_length(self) -> "ChatResponse":
        """Enforce recommendation count rules.

        The schema must allow an empty list during clarification/refusal, but if
        recommendations are present they must be between 1 and 10 items.
        """
        if self.recommendations and not (1 <= len(self.recommendations) <= 10):
            raise ValueError("recommendations must contain between 1 and 10 items")
        return self


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: Literal["ok"]


class CatalogItem(BaseModel):
    """Clean internal representation of one SHL catalog item."""

    name: str
    url: HttpUrl
    description: str = ""
    test_type: str = ""
    keys: list[str] = Field(default_factory=list)
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration: str = ""
    searchable_text: str = ""
