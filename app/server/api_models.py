from pydantic import BaseModel

from app.data.dtos.fact import FactDTO


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


class ExtractFactsRequest(BaseModel):
    user_id: str
    trip_id: str
    limit: int = 5


class ExtractFactsResponse(BaseModel):
    extracted_facts: list[FactDTO]


class GetAllFactsRequest(BaseModel):
    user_id: str


class GetAllFactsResponse(BaseModel):
    facts: list[FactDTO]
