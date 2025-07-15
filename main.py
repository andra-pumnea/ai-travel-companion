import logging
import uvicorn
from fastapi import FastAPI

from app.core.settings import APISettings
from app.server.routers import journal, planner, user_facts, chat


def setup_logging():
    """Configure basic logging for the application."""
    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger().setLevel(logging.INFO)


def create_app() -> FastAPI:
    app_settings = APISettings()
    app = FastAPI(
        title=app_settings.project_name,
        description=app_settings.project_description,
        version=app_settings.project_version,
    )

    app.include_router(
        chat.router,
        prefix="/chat",
        tags=["Chat Assistant"]
    )

    app.include_router(
        journal.router,
        prefix="/journal",
        tags=["Travel Journal"],
    )

    app.include_router(
        planner.router,
        prefix="/planner",
        tags=["Trip Planning"],
    )

    app.include_router(
        user_facts.router,
        prefix="/user_facts",
        tags=["User Facts"],
    )

    return app


app = create_app()

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting Travel Journal RAG Assistant API")
    uvicorn.run(app, port=8000, host="0.0.0.0", log_level="info")
