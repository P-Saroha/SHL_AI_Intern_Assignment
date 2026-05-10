from fastapi import FastAPI

from app.config import settings
from app.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    FastAPI is the web framework. It receives HTTP requests, validates them,
    calls our route functions, and sends JSON responses back to the client.

    Keeping app creation in a function makes the project easier to test later.
    """
    api = FastAPI(title=settings.app_name)
    # Modular routing keeps endpoint definitions separate from app startup.
    api.include_router(router)
    return api


# Uvicorn looks for this variable when we run:
# uvicorn app.main:app --reload
app = create_app()
