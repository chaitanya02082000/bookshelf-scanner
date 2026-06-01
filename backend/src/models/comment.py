from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel


class BookCommentBookInput(BaseModel):
    book_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    isbn: str | None = None

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }


class BookCommentListRequest(BookCommentBookInput):
    pass


class CreateBookCommentRequest(BookCommentBookInput):
    body: str = Field(min_length=1, max_length=1000)
    user_display_name: str | None = None


class BookComment(BaseModel):
    id: str
    book_key: str
    book_id: str
    title: str
    authors: list[str]
    isbn: str | None = None
    user_display_name: str
    body: str
    created_at: str
    updated_at: str
    is_owner: bool = False

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }
