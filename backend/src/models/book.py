from pydantic import BaseModel
from pydantic.alias_generators import to_camel


class BookMetadata(BaseModel):
    id: str
    title: str
    authors: list[str]
    cover_url: str | None = None
    description: str | None = None
    summary: str | None = None
    subjects: list[str] | None = None
    published_date: str | None = None
    page_count: int | None = None
    isbn: str | None = None
    source: str | None = None

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }
