from pydantic import Field
from pydantic.alias_generators import to_camel

from .book import BookMetadata


class BookRecommendation(BookMetadata):
    score: float
    reason: str | None = None
    matched_authors: list[str] = Field(default_factory=list)
    matched_subjects: list[str] = Field(default_factory=list)
    matched_queries: list[str] = Field(default_factory=list)

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }
