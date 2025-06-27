import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.data.models.trip import TripDTO


def read_trip_from_polarsteps() -> TripDTO:
    with open(
        os.path.expanduser("~/ai-travel-companion/app/data/files/trip.json"),
        "r",
    ) as f:
        data = json.load(f)
    trip = TripDTO.from_raw_json(data)
    return trip
