from pydantic import BaseModel
from typing import Optional


class Location(BaseModel):
    lat: float
    lon: float
    name: str
    detail: str
    country_code: str


class TripStep(BaseModel):
    id: int
    display_name: str
    description: Optional[str]
    location: Location


class Trip(BaseModel):
    all_steps: list[TripStep]
