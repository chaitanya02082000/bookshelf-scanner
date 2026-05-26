from pydantic import Field
from pydantic.alias_generators import to_camel

from .book import BookMetadata


class BookRecommendation(BookMetadata):
    score: float
    embedding_score: float = 0
    content_score: float = 0
    collaborative_score: float = 0
    reason: str | None = None
    matched_authors: list[str] = Field(default_factory=list)
    matched_subjects: list[str] = Field(default_factory=list)
    matched_queries: list[str] = Field(default_factory=list)
    matched_books: list[str] = Field(default_factory=list)

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }
