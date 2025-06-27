from pydantic import BaseModel


class SearchJournalRequest(BaseModel):
    user_query: str
    user_id: str
    trip_id: str
    limit: int = 5


class SearchJournalResponse(BaseModel):
    documents: list[dict]


class SearchJournalWithGenerationRequest(BaseModel):
    user_query: str
    user_id: str
    trip_id: str
    limit: int = 5


class SearchJournalWithGenerationResponse(BaseModel):
    answer: str
    documents: list[dict]


class PlanTripRequest(BaseModel):
    user_query: str
    user_trip_id: str
    max_steps: int = 5


class PlanTripResponse(BaseModel):
    answer: str
    thought_process: str
