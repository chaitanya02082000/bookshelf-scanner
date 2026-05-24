from __future__ import annotations

from libgen_api_enhanced import LibgenSearch

from src.models.libgen import LibgenBook


class LibgenService:
    def __init__(self, mirror: str = "li") -> None:
        self.client = LibgenSearch(mirror=mirror)

    def search(self, query: str, limit: int = 8) -> list[LibgenBook]:
        if not query.strip():
            return []

        results = self.client.search_default(query)
        books = [self._map_book(book) for book in results[:limit]]
        return books

    def resolve_download_link(
        self, query: str, title: str, author: str | None = None, md5: str | None = None
    ) -> str | None:
        candidates = self.client.search_default(query)
        for book in candidates:
            if md5 and getattr(book, "md5", None) == md5:
                book.resolve_direct_download_link()
                return getattr(book, "resolved_download_link", None)

        normalized_title = self._normalize(title)
        normalized_author = self._normalize(author or "")
        for book in candidates:
            book_title = self._normalize(getattr(book, "title", ""))
            book_author = self._normalize(getattr(book, "author", ""))
            if book_title == normalized_title and (
                not normalized_author or book_author == normalized_author
            ):
                book.resolve_direct_download_link()
                return getattr(book, "resolved_download_link", None)
        return None

    def _map_book(self, book: object) -> LibgenBook:
        return LibgenBook(
            id=str(getattr(book, "id", "")),
            title=str(getattr(book, "title", "Untitled")),
            author=self._optional_text(getattr(book, "author", None)),
            publisher=self._optional_text(getattr(book, "publisher", None)),
            year=self._optional_text(getattr(book, "year", None)),
            language=self._optional_text(getattr(book, "language", None)),
            pages=self._optional_text(getattr(book, "pages", None)),
            size=self._optional_text(getattr(book, "size", None)),
            extension=self._optional_text(getattr(book, "extension", None)),
            md5=self._optional_text(getattr(book, "md5", None)),
            mirrors=list(getattr(book, "mirrors", []) or []),
            tor_download_link=self._optional_text(
                getattr(book, "tor_download_link", None)
            ),
            resolved_download_link=self._optional_text(
                getattr(book, "resolved_download_link", None)
            ),
        )

    def _optional_text(self, value: object | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().split())
