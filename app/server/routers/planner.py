from fastapi import APIRouter, status, HTTPException

from app.server.api_models import PlanTripRequest, PlanTripResponse
from app.services.planner_service import PlannerService

router = APIRouter()
planner_service = PlannerService()


@router.post(
    "/plan_trip",
    response_model=PlanTripResponse,
    status_code=status.HTTP_200_OK,
)
async def plan_trip(request: PlanTripRequest) -> PlanTripResponse:
    """
    Endpoint to plan a trip based on the user's query and trip ID.
    :param request: PlanTripRequest containing user query, user trip ID, and max steps.
    :return: PlanTripResponse containing the planned trip details.
    """

    try:
        response = await planner_service.plan_trip(
            user_query=request.user_query,
            user_id=request.user_id,
            trip_id=request.trip_id,
            max_steps=request.max_steps,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
    return PlanTripResponse(answer=response.answer)
