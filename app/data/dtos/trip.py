from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class Location(BaseModel):
    lat: float
    lon: float
    name: str
    detail: str
    country_code: str


class TripStepDTO(BaseModel):
    id: int
    display_name: str
    description: Optional[str]
    location_name: str
    lat: float
    lon: float
    detail: str
    country_code: str
    weather_condition: Optional[str] = None
    weather_temperature: Optional[float] = None

    @classmethod
    def from_raw_json(cls, data: dict) -> TripStepDTO:
        location = data.get("location", {})
        return cls(
            id=data["id"],
            display_name=data.get("display_name") or location.get("name") or "Unknown",
            description=data.get("description"),
            location_name=location.get("name") or "Unknown",
            lat=location.get("lat", 0.0),
            lon=location.get("lon", 0.0),
            detail=location.get("detail") or location.get("full_detail") or "",
            country_code=location.get("country_code") or "XX",
            weather_condition=data.get("weather_condition"),
            weather_temperature=data.get("weather_temperature"),
        )


class TripDTO(BaseModel):
    id: int
    user_id: int
    name: str
    summary: Optional[str]
    all_steps: list[TripStepDTO]

    @classmethod
    def from_raw_json(cls, data: dict) -> "TripDTO":
        steps = data.get("all_steps", [])
        parsed_steps = [TripStepDTO.from_raw_json(step) for step in steps]
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            name=data["name"],
            summary=data.get("summary"),
            all_steps=parsed_steps,
        )
