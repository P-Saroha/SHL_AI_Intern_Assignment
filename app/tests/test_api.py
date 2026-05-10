from fastapi.testclient import TestClient

from app.main import app
from app.models import ChatResponse
from app import routes


class FakeAgent:
    def respond(self, request):
        return ChatResponse(
            reply="ok",
            recommendations=[],
            end_of_conversation=False,
        )


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_valid_request():
    routes.agent = FakeAgent()
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Hiring Java developer"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert payload["recommendations"] == []


def test_chat_endpoint_malformed_request():
    client = TestClient(app)
    response = client.post("/chat", json={"messages": []})
    assert response.status_code == 422
