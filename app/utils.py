import re
from pathlib import Path

from app.models import CatalogItem, Message


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

GENERIC_TOKENS = {
    "need",
    "assessment",
    "assessments",
    "test",
    "tests",
    "hire",
    "hiring",
    "looking",
    "role",
    "candidate",
}


def normalize_text(text: str) -> str:
    """Lowercase text and remove punctuation for simple matching."""
    return re.sub(r"[^a-z0-9+#.]+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    """Split text into searchable tokens.

    Tokens are used for keyword overlap scoring in hybrid retrieval.
    """
    return [
        token
        for token in normalize_text(text).split()
        if token and token not in STOPWORDS
    ]


def format_conversation(messages: list[Message]) -> str:
    """Render conversation history as plain text for prompting."""
    lines: list[str] = []
    for message in messages:
        role = "User" if message.role == "user" else "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


def format_catalog_context(items: list[CatalogItem]) -> str:
    """Build a grounded catalog context block for the LLM.

    We keep only catalog fields to prevent the model from inventing details.
    """
    lines: list[str] = []
    for item in items:
        keys = ", ".join(item.keys) or "Not specified"
        levels = ", ".join(item.job_levels) or "Not specified"
        languages = ", ".join(item.languages) or "Not specified"
        duration = item.duration or "Not specified"
        lines.append(
            " | ".join(
                [
                    f"Name: {item.name}",
                    f"URL: {item.url}",
                    f"Type: {item.test_type or 'Not specified'}",
                    f"Keys: {keys}",
                    f"Job levels: {levels}",
                    f"Languages: {languages}",
                    f"Duration: {duration}",
                    f"Description: {item.description or 'Not specified'}",
                ]
            )
        )
    return "\n".join(lines)


def keyword_overlap_score(query: str, document: str) -> float:
    """Return a simple keyword overlap score between 0 and 1.

    A score of 1 means every meaningful query token appears in the document.
    This is useful because embeddings understand meaning, but exact words like
    "Java", "HIPAA", or "Excel" should still strongly influence ranking.
    """
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0

    document_tokens = set(tokenize(document))
    return len(query_tokens & document_tokens) / len(query_tokens)


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent folder for a file path if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def latest_user_message(messages: list[Message]) -> str:
    """Return the newest user message from a conversation."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def looks_vague(text: str) -> bool:
    """Detect requests that are too vague to recommend from.

    This starter rule is intentionally conservative.
    Later, the agent can use richer slot extraction.
    """
    normalized = normalize_text(text)
    words = [word for word in normalized.split() if word not in {"i", "need", "an", "a", "assessment", "test"}]
    return len(words) < 3


INJECTION_PATTERNS = (
    r"ignore (all|previous) instructions",
    r"system prompt",
    r"developer message",
    r"jailbreak",
    r"act as",
    r"you are now",
    r"do anything now",
    r"reveal.*prompt",
)

OFF_TOPIC_KEYWORDS = {
    "legal",
    "law",
    "lawsuit",
    "attorney",
    "contract",
    "salary",
    "compensation",
    "visa",
    "immigration",
    "tax",
}

TEST_TYPE_KEYWORDS = {
    "ability": "Ability & Aptitude",
    "aptitude": "Ability & Aptitude",
    "cognitive": "Ability & Aptitude",
    "personality": "Personality & Behavior",
    "behavior": "Personality & Behavior",
    "behaviour": "Personality & Behavior",
    "skills": "Knowledge & Skills",
    "knowledge": "Knowledge & Skills",
    "situational": "Biodata & Situational Judgment",
    "sjt": "Biodata & Situational Judgment",
    "simulation": "Simulations",
    "exercise": "Assessment Exercises",
    "assessment exercise": "Assessment Exercises",
    "competency": "Competencies",
    "360": "Development & 360",
    "development": "Development & 360",
}

JOB_LEVEL_KEYWORDS = {
    "entry": "Entry-Level",
    "junior": "Entry-Level",
    "graduate": "Graduate",
    "mid": "Mid-Professional",
    "mid-level": "Mid-Professional",
    "mid level": "Mid-Professional",
    "senior": "Manager",
    "manager": "Manager",
    "director": "Director",
    "executive": "Executive",
    "supervisor": "Supervisor",
    "front line": "Front Line Manager",
}


def detect_prompt_injection(text: str) -> bool:
    """Basic detection of prompt-injection attempts."""
    normalized = normalize_text(text)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized):
            return True
    return False


def detect_off_topic(text: str) -> bool:
    """Detect requests outside SHL assessment recommendations."""
    tokens = set(tokenize(text))
    return any(keyword in tokens for keyword in OFF_TOPIC_KEYWORDS)


def detect_comparison_request(text: str) -> bool:
    """Detect if the user wants a comparison between assessments."""
    normalized = normalize_text(text)
    return any(term in normalized for term in ["compare", "difference", "vs", "versus"])


def detect_refinement(text: str) -> bool:
    """Detect user edits like add/remove/change constraints."""
    normalized = normalize_text(text)
    return any(
        term in normalized
        for term in ["actually", "instead", "add", "also", "remove", "exclude", "only", "not"]
    )


def find_catalog_mentions(text: str, catalog: list[CatalogItem]) -> list[CatalogItem]:
    """Find catalog items whose names appear in user text.

    This prevents hallucinations: we only compare items that exist in the catalog.
    """
    normalized_text = normalize_text(text)
    matches: list[CatalogItem] = []
    for item in catalog:
        name = normalize_text(item.name)
        if name and name in normalized_text:
            matches.append(item)
    return matches


def extract_constraints(messages: list[Message]) -> dict[str, list[str]]:
    """Extract simple constraints from the full conversation history.

    Stateless APIs must reconstruct context on every call. We do that by
    scanning all user messages for role, level, language, and test-type cues.
    """
    role_terms: list[str] = []
    skill_terms: list[str] = []
    job_levels: list[str] = []
    languages: list[str] = []
    test_types: list[str] = []

    for message in messages:
        if message.role != "user":
            continue
        text = normalize_text(message.content)
        tokens = tokenize(text)

        for token in tokens:
            if token in JOB_LEVEL_KEYWORDS:
                job_levels.append(JOB_LEVEL_KEYWORDS[token])
            if token in TEST_TYPE_KEYWORDS:
                test_types.append(TEST_TYPE_KEYWORDS[token])
            if token in {"english", "spanish", "french", "german", "arabic", "hindi"}:
                languages.append(token.title())

        # Heuristic: treat longer tokens as role or skill hints.
        for token in tokens:
            if len(token) >= 4 and token not in OFF_TOPIC_KEYWORDS and token not in GENERIC_TOKENS:
                role_terms.append(token)

        for token in tokens:
            if len(token) >= 3 and token not in OFF_TOPIC_KEYWORDS and token not in GENERIC_TOKENS:
                skill_terms.append(token)

    # De-duplicate while preserving order.
    def dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if value not in seen:
                ordered.append(value)
                seen.add(value)
        return ordered

    return {
        "role_terms": dedupe(role_terms),
        "skill_terms": dedupe(skill_terms),
        "job_levels": dedupe(job_levels),
        "languages": dedupe(languages),
        "test_types": dedupe(test_types),
    }


def build_query_from_constraints(constraints: dict[str, list[str]]) -> str:
    """Build a single search query for retrieval."""
    parts: list[str] = []
    parts.extend(constraints.get("role_terms", []))
    parts.extend(constraints.get("skill_terms", []))
    parts.extend(constraints.get("job_levels", []))
    parts.extend(constraints.get("languages", []))
    parts.extend(constraints.get("test_types", []))
    return " ".join(parts)


def is_vague_request(constraints: dict[str, list[str]]) -> bool:
    """Return True when the conversation lacks usable constraints."""
    role_terms = constraints.get("role_terms", [])
    skill_terms = constraints.get("skill_terms", [])
    test_types = constraints.get("test_types", [])
    return len(role_terms) < 2 and len(skill_terms) < 3 and not test_types


def has_minimum_context(constraints: dict[str, list[str]]) -> bool:
    """Decide if we have enough context to recommend from the catalog."""
    role_terms = constraints.get("role_terms", [])
    skill_terms = constraints.get("skill_terms", [])
    test_types = constraints.get("test_types", [])
    return len(role_terms) >= 2 or len(skill_terms) >= 3 or bool(test_types)
