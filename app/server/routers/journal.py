from fastapi import APIRouter, HTTPException, status

from app.server.api_models import (
    SearchJournalRequest,
    SearchJournalResponse,
    SearchJournalWithGenerationRequest,
    SearchJournalWithGenerationResponse,
)
from app.services.journal_service import JournalService
from app.exceptions import CollectionNotFoundError, LLMManagerError


router = APIRouter()
journal_service = JournalService()


@router.post(
    "/search", response_model=SearchJournalResponse, status_code=status.HTTP_200_OK
)
async def search_journal(request: SearchJournalRequest) -> SearchJournalResponse:
    """
    Endpoint to search the travel journal based on user query, user ID, and trip ID.
    :param request: SearchJournalRequest containing user query, user ID, trip ID, and limit.
    :return: SearchJournalResponse containing a list of documents matching the query.
    """

    try:
        documents = await journal_service.search_journal(
            user_query=request.user_query,
            user_id=request.user_id,
            trip_id=request.trip_id,
            limit=request.limit,
        )
    except CollectionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
    return SearchJournalResponse(documents=documents)


@router.post(
    "/search_with_generation",
    response_model=SearchJournalWithGenerationResponse,
    status_code=status.HTTP_200_OK,
)
async def search_journal_with_generation(
    request: SearchJournalWithGenerationRequest,
) -> SearchJournalWithGenerationResponse:
    """
    Endpoint to search the travel journal using RAG.
    :param request: SearchJournalWithGenerationRequest containing user query, user ID, trip ID, and limit.
    :return: SearchJournalWithGenerationResponse containing a list of documents matching the query and generated answer.
    """

    try:
        answer, documents = await journal_service.search_journal_with_generation(
            user_query=request.user_query,
            user_id=request.user_id,
            trip_id=request.trip_id,
            limit=request.limit,
        )
    except CollectionNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except LLMManagerError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )

    return SearchJournalWithGenerationResponse(answer=answer, documents=documents)
