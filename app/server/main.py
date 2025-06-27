import sys
import os
import logging
import uvicorn
from fastapi import FastAPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from app.server.router import api_router


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


if __name__ == "__main__":
    setup_logging()
    logging.info("Starting Travel Journal RAG Assistant API")

    app = FastAPI(
        title="Travel Journal RAG Assistant API",
        description="API for the Travel Journal RAG Assistant, providing endpoints to search travel journals.",
        version="1.0.0",
    )
    app.include_router(api_router)
    uvicorn.run(app, log_level="info")
