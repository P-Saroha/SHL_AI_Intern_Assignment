import pytest
from pydantic import ValidationError

from app.models import ChatResponse, Recommendation


def test_recommendations_length_validation():
    recommendations = [
        Recommendation(name=f"Test {i}", url="https://example.com", test_type="K")
        for i in range(11)
    ]

    with pytest.raises(ValidationError):
        ChatResponse(reply="ok", recommendations=recommendations, end_of_conversation=False)
