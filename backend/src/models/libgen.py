from pydantic import BaseModel
from pydantic.alias_generators import to_camel


class LibgenBook(BaseModel):
    id: str
    title: str
    author: str | None = None
    publisher: str | None = None
    year: str | None = None
    language: str | None = None
    pages: str | None = None
    size: str | None = None
    extension: str | None = None
    md5: str | None = None
    mirrors: list[str] = []
    tor_download_link: str | None = None
    resolved_download_link: str | None = None
    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }


class LibgenResolveRequest(BaseModel):
    query: str
    md5: str | None = None
    title: str
    author: str | None = None
    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }
