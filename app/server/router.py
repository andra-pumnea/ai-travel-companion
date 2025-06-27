from fastapi import APIRouter
import app.server.endpoints.journal_endpoints as journal_endpoints
import app.server.endpoints.planner_endpoints as planner_endpoints

api_router = APIRouter()

api_router.include_router(
    journal_endpoints.router,
    prefix="/journal",
    tags=["Travel Journal"],
)

api_router.include_router(
    planner_endpoints.router,
    prefix="/planner",
    tags=["Trip Planning"],
)
