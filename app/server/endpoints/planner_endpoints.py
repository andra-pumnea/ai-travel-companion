from app.server.api_models import PlanTripRequest, PlanTripResponse
from app.services.planner_service import PlannerService
from fastapi import APIRouter
from http import HTTPStatus

router = APIRouter()
planner_service = PlannerService()


@router.post(
    "/plan_trip",
    response_model=PlanTripResponse,
    status_code=HTTPStatus.OK,
)
async def plan_trip(request: PlanTripRequest) -> PlanTripResponse:
    """
    Endpoint to plan a trip based on the user's query and trip ID.
    :param request: PlanTripRequest containing user query, user trip ID, and max steps.
    :return: PlanTripResponse containing the planned trip details.
    """

    response = await planner_service.plan_trip(
        user_query=request.user_query,
        user_trip_id=request.user_trip_id,
        max_steps=request.max_steps,
    )

    return PlanTripResponse(
        answer=response.answer, thought_process=response.thought_process
    )
