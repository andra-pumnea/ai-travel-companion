import logging

from fastapi import APIRouter, status, HTTPException

from app.services.long_term_memory_service import FactService
from app.server.api_models import ExtractFactsRequest, ExtractFactsResponse
from app.core.exceptions.llm_exceptions import LLMManagerError

router = APIRouter()
fact_service = FactService()


@router.post(
    "/extract_facts",
    response_model=ExtractFactsResponse,
    status_code=status.HTTP_200_OK,
)
async def extract_facts(request: ExtractFactsRequest) -> ExtractFactsResponse:
    """
    Endpoint to extract facts from the user's travel journal.
    :param request: ExtractFactsRequest containing user ID, trip ID, and limit.
    :return: ExtractFactsResponse containing the extracted facts.
    """

    try:
        facts = await fact_service.extract_facts(
            user_id=request.user_id, trip_id=request.trip_id
        )
    except LLMManagerError as e:
        logging.error(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while extracting facts: {str(e)}",
        )
    except Exception as e:
        logging.error(f"Error extracting facts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while extracting facts: {str(e)}",
        )

    return ExtractFactsResponse(
        thought_process=facts.thought_process,
        extracted_facts=facts.extracted_facts,
    )
