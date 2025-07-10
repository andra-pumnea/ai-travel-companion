import logging

from fastapi import APIRouter, status, HTTPException, Depends

from app.services.facts_service import FactService
from app.server.api_models import (
    ExtractFactsRequest,
    ExtractFactsResponse,
    GetAllFactsResponse,
)
from app.core.exceptions.llm_exceptions import LLMManagerError
from app.server.dependencies import get_fact_service

router = APIRouter()


@router.post(
    "/extract_facts",
    response_model=ExtractFactsResponse,
    status_code=status.HTTP_200_OK,
)
async def extract_facts(
    request: ExtractFactsRequest, fact_service: FactService = Depends(get_fact_service)
) -> ExtractFactsResponse:
    """
    Endpoint to extract facts from the user's travel journal.
    :param request: ExtractFactsRequest containing user ID, trip ID, and limit.
    :return: ExtractFactsResponse containing the extracted facts.
    """

    try:
        facts = await fact_service.extract_facts(
            user_id=request.user_id, trip_id=request.trip_id, limit=request.limit
        )
    except LLMManagerError as e:
        logging.error(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while extracting facts: {str(e)}",
        )
    except Exception as e:
        logging.error(f"Error extracting facts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while extracting facts: {str(e)}",
        )

    return ExtractFactsResponse(
        extracted_facts=facts,
    )


@router.get(
    "/{user_id}",
    response_model=GetAllFactsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_all_user_facts(
    user_id: str, fact_service: FactService = Depends(get_fact_service)
) -> GetAllFactsResponse:
    """
    Endpoint to retrieve all facts for a user.
    :param user_id: The ID of the user to retrieve facts for.
    :return: GetAllFactsResponse containing the user's facts.
    """
    try:
        facts = await fact_service.get_all_facts(user_id=user_id)
    except Exception as e:
        logging.error(f"Error retrieving facts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving facts: {str(e)}",
        )

    return GetAllFactsResponse(
        facts=facts,
    )
