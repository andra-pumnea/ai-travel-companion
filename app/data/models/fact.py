from pydantic import BaseModel, Field


class FactDTO(BaseModel):
    """
    Model representing a single fact extracted from the user's travel journal.
    """

    fact_text: str = Field(description="The text of the extracted fact.")
    category: str = Field(
        description="The category of the fact, such as 'food', 'activity', 'travel_style', 'transportation', 'budget'",
    )
