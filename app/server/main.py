import logging

from fastapi import FastAPI
from app.server.router import api_router


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


setup_logging
app = FastAPI(
    title="Travel Journal RAG Assistant API",
    description="API for the Travel Journal RAG Assistant, providing endpoints to search travel journals.",
    version="1.0.0",
)
app.include_router(api_router)
