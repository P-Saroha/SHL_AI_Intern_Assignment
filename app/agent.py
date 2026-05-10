from app.catalog_loader import CatalogLoader
from app.config import settings
from app.models import CatalogItem, ChatRequest, ChatResponse
from app.prompts import (
    CLARIFICATION_TEMPLATE,
    COMPARISON_INTRO,
    COMPARISON_REPLY_PROMPT,
    RECOMMENDATION_INTRO,
    RECOMMENDATION_REPLY_PROMPT,
    REFINEMENT_INTRO,
    REFUSAL_TEMPLATE,
)
from app.retriever import HybridRetriever
from app.utils import (
    build_query_from_constraints,
    detect_comparison_request,
    detect_off_topic,
    detect_prompt_injection,
    detect_refinement,
    extract_constraints,
    find_catalog_mentions,
    format_catalog_context,
    format_conversation,
    has_minimum_context,
    is_vague_request,
    latest_user_message,
)


class GeminiClient:
    """Minimal Gemini wrapper with safe defaults.

    This client only generates conversational text. It never selects or
    fabricates assessments. The retriever remains the source of truth.
    """

    def __init__(self, api_key: str, model_name: str, temperature: float, max_tokens: int) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def generate(self, prompt: str) -> str | None:
        if not self.api_key:
            return None

        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                # If the dependency is missing, fall back to deterministic replies.
                return None

            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_name)

        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
        except Exception:
            # Gemini failures should not break the API response.
            return None

        text = getattr(response, "text", "") or ""
        return text.strip() or None


class RecommendationAgent:
    """Coordinates the conversation.

    This is the place where the final system will decide:
    - Should we ask a clarification question?
    - Should we retrieve assessments?
    - Should we compare assessments?
    - Should we refine an existing shortlist?
    - Should we refuse an out-of-scope request?
    """

    def __init__(self) -> None:
        # Load the catalog once. Requests are stateless, but the service can
        # cache shared read-only data safely at process startup.
        self.catalog = CatalogLoader(settings.catalog_path).load()
        self.catalog_by_url = {str(item.url): item for item in self.catalog}
        self.retriever = HybridRetriever(self.catalog)
        self.gemini = GeminiClient(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
            temperature=settings.gemini_temperature,
            max_tokens=settings.gemini_max_output_tokens,
        )

    def respond(self, request: ChatRequest) -> ChatResponse:
        """Create the next assistant response.

        Starter behavior:
        - Ask a clarification question.
        - Keep recommendations empty until the real retrieval and Gemini logic
          is implemented later.

        Important: this method receives the full history in `request`.
        It should not rely on saved session state.
        """
        user_text = latest_user_message(request.messages)

        # Refuse off-topic or prompt-injection attempts.
        if detect_prompt_injection(user_text) or detect_off_topic(user_text):
            return ChatResponse(reply=REFUSAL_TEMPLATE, recommendations=[], end_of_conversation=False)

        # Comparison requests must be answered using catalog data only.
        if detect_comparison_request(user_text):
            return self._handle_comparison(request, user_text)

        # Extract constraints from full history because the API is stateless.
        constraints = extract_constraints(request.messages)

        # If still vague, ask a targeted clarification question.
        # Delaying recommendations here improves quality and avoids guessing.
        if is_vague_request(constraints) or not has_minimum_context(constraints):
            return ChatResponse(reply=CLARIFICATION_TEMPLATE, recommendations=[], end_of_conversation=False)

        # Build a retrieval query that aggregates role, skills, level, and test type hints.
        query = build_query_from_constraints(constraints)

        results = self.retriever.search(query, top_k=settings.max_recommendations)
        if not results:
            return ChatResponse(
                reply=(
                    "I could not find matching assessments in the SHL catalog yet. "
                    "Could you share the role, level, or specific skills to measure?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        intro = REFINEMENT_INTRO if detect_refinement(user_text) else RECOMMENDATION_INTRO
        reply = self._generate_recommendation_reply(request, results, intro)
        return ChatResponse(reply=reply, recommendations=results, end_of_conversation=False)

    def _handle_comparison(self, request: ChatRequest, user_text: str) -> ChatResponse:
        """Compare two assessments using catalog data only.

        This prevents hallucinations by relying on data already scraped from SHL.
        """
        matches = find_catalog_mentions(user_text, self.catalog)
        unique: list[CatalogItem] = []
        seen: set[str] = set()
        for item in matches:
            if item.name not in seen:
                unique.append(item)
                seen.add(item.name)

        if len(unique) < 2:
            return ChatResponse(
                reply=(
                    "Which two SHL assessments should I compare? "
                    "For example, 'Compare OPQ32r and GSA'."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # Compare the first two unique matches mentioned.
        left, right = unique[0], unique[1]
        reply = self._format_comparison(request, left, right)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    def _format_comparison(self, request: ChatRequest, left: CatalogItem, right: CatalogItem) -> str:
        """Format a side-by-side comparison using catalog fields only."""
        # The comparison prompt is grounded in the exact catalog fields.
        catalog_context = format_catalog_context([left, right])
        conversation = format_conversation(request.messages)
        prompt = COMPARISON_REPLY_PROMPT.format(
            conversation=conversation,
            catalog_context=catalog_context,
        )

        response = self.gemini.generate(prompt)
        if response:
            # Guardrail: if Gemini mentions any other catalog item, fall back.
            mentions = find_catalog_mentions(response, self.catalog)
            allowed = {left.name, right.name}
            if all(item.name in allowed for item in mentions):
                return response

        left_keys = ", ".join(left.keys) or "Not specified"
        right_keys = ", ".join(right.keys) or "Not specified"
        left_levels = ", ".join(left.job_levels) or "Not specified"
        right_levels = ", ".join(right.job_levels) or "Not specified"
        left_lang = ", ".join(left.languages) or "Not specified"
        right_lang = ", ".join(right.languages) or "Not specified"

        return (
            f"{COMPARISON_INTRO}\n"
            f"- {left.name}: {left.description or 'No catalog description.'}\n"
            f"  Focus: {left_keys}. Job levels: {left_levels}. Languages: {left_lang}.\n"
            f"- {right.name}: {right.description or 'No catalog description.'}\n"
            f"  Focus: {right_keys}. Job levels: {right_levels}. Languages: {right_lang}."
        )

    def _generate_recommendation_reply(
        self, request: ChatRequest, results: list[dict[str, str]], intro: str
    ) -> str:
        """Generate a grounded reply for the current recommendation list."""
        # The LLM only writes a summary. It does not choose assessments.
        catalog_items: list[CatalogItem] = []
        for item in results:
            catalog_item = self.catalog_by_url.get(item["url"])
            if catalog_item:
                catalog_items.append(catalog_item)

        if not catalog_items:
            return intro

        conversation = format_conversation(request.messages)
        catalog_context = format_catalog_context(catalog_items)
        prompt = RECOMMENDATION_REPLY_PROMPT.format(
            conversation=conversation,
            catalog_context=catalog_context,
        )

        response = self.gemini.generate(prompt)
        if response:
            # Guardrail: summaries must not name assessments.
            if not find_catalog_mentions(response, self.catalog):
                return response

        # Fallback: minimal deterministic response if Gemini is unavailable.
        return intro
