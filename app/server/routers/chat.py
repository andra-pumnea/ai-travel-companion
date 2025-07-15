from fastapi import APIRouter, status, HTTPException

from app.services.chat_service import ChatService
from app.server.api_models import ReplyReponse, ReplyRequest

router = APIRouter()
chat_service = ChatService()

@router.post(
    "/reply",
    response_model=ReplyReponse,
    status_code=status.HTTP_200_OK,
)
async def reply(request: ReplyRequest) -> ReplyReponse:

    try:
        response = await chat_service.reply(
            user_query=request.user_query,
            user_id=request.user_id,
            trip_id=request.trip_id,
            conversation_id=request.conversation_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
    return ReplyReponse(answer=response)