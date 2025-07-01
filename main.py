import logging
import uvicorn
from fastapi import FastAPI

from app.core.settings import APISettings
from app.server.routers import journal, planner


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_app() -> FastAPI:
    app_settings = APISettings()
    app = FastAPI(
        title=app_settings.project_name,
        description=app_settings.project_description,
        version=app_settings.project_version,
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

    return app


app = create_app()

if __name__ == "__main__":
    setup_logging()
    logging.info("Starting Travel Journal RAG Assistant API")
    uvicorn.run(app, log_level="info")
