from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel


class SearchHistoryEventInput(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    source: str | None = None
    selected_book_id: str | None = None
    selected_title: str | None = None
    selected_authors: list[str] | None = None
    selected_subjects: list[str] | None = None

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }


class SearchHistoryEvent(SearchHistoryEventInput):
    normalized_query: str
    created_at: str

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }
