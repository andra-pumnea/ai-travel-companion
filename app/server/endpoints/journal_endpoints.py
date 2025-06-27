from fastapi import APIRouter
from http import HTTPStatus

from app.server.api_models import (
    SearchJournalRequest,
    SearchJournalResponse,
    SearchJournalWithGenerationRequest,
    SearchJournalWithGenerationResponse,
)
from app.services.journal_service import JournalService


router = APIRouter()
journal_service = JournalService()


@router.post("/search", response_model=SearchJournalResponse, status_code=HTTPStatus.OK)
async def search_journal(request: SearchJournalRequest) -> SearchJournalResponse:
    """
    Endpoint to search the travel journal based on user query, user ID, and trip ID.
    :param request: SearchJournalRequest containing user query, user ID, trip ID, and limit.
    :return: SearchJournalResponse containing a list of documents matching the query.
    """

    documents = await journal_service.search_journal(
        user_query=request.user_query,
        user_id=request.user_id,
        trip_id=request.trip_id,
        limit=request.limit,
    )

    return SearchJournalResponse(documents=documents)


@router.post(
    "/search_with_generation",
    response_model=SearchJournalWithGenerationResponse,
    status_code=HTTPStatus.OK,
)
async def search_journal_with_generation(
    request: SearchJournalWithGenerationRequest,
) -> SearchJournalWithGenerationResponse:
    """
    Endpoint to search the travel journal using RAG.
    :param request: SearchJournalWithGenerationRequest containing user query, user ID, trip ID, and limit.
    :return: SearchJournalWithGenerationResponse containing a list of documents matching the query and generated answer.
    """

    answer, documents = await journal_service.search_journal_with_generation(
        user_query=request.user_query,
        user_id=request.user_id,
        trip_id=request.trip_id,
        limit=request.limit,
    )

    return SearchJournalWithGenerationResponse(answer=answer, documents=documents)
