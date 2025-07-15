from pydantic import BaseModel

from app.data.dtos.fact import FactDTO


class BaseUserRequest(BaseModel):
    user_id: str
    trip_id: str


class SearchJournalRequest(BaseUserRequest):
    user_query: str
    limit: int = 5


class SearchJournalResponse(BaseModel):
    documents: list[dict]


class SearchJournalWithGenerationRequest(BaseUserRequest):
    user_query: str
    limit: int = 5


class SearchJournalWithGenerationResponse(BaseModel):
    answer: str
    documents: list[dict]


class PlanTripRequest(BaseUserRequest):
    user_query: str
    max_steps: int = 5


class PlanTripResponse(BaseModel):
    answer: str


class ExtractFactsRequest(BaseUserRequest):
    limit: int = 5


class ExtractFactsResponse(BaseModel):
    extracted_facts: list[FactDTO]


class GetAllFactsRequest(BaseModel):
    user_id: str


class GetAllFactsResponse(BaseModel):
    facts: list[FactDTO]
