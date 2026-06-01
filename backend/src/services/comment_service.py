from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import re

from src.models import BookComment, BookCommentListRequest, CreateBookCommentRequest
from src.services.auth_service import AuthenticatedUser
from src.services.mongo_service import mongo_service


class CommentService:
    def list_comments(
        self,
        user: AuthenticatedUser,
        request: BookCommentListRequest,
        limit: int = 100,
    ) -> list[BookComment]:
        collection = mongo_service.get_comments_collection()
        docs = (
            collection.find(
                {"bookKey": self._book_key(request.book_id, request.isbn, request.title, request.authors)},
                {"_id": 0},
            )
            .sort("createdAt", -1)
            .limit(limit)
        )
        return [self._to_model(user, doc) for doc in docs]

    def create_comment(
        self, user: AuthenticatedUser, request: CreateBookCommentRequest
    ) -> BookComment:
        collection = mongo_service.get_comments_collection()
        now = datetime.now(timezone.utc).isoformat()
        comment_id = uuid4().hex
        book_key = self._book_key(
            request.book_id, request.isbn, request.title, request.authors
        )
        doc = {
            "id": comment_id,
            "bookKey": book_key,
            "bookId": request.book_id,
            "title": request.title,
            "authors": request.authors,
            "isbn": request.isbn,
            "auth0UserId": user.auth0_user_id,
            "userDisplayName": self._display_name(user, request.user_display_name),
            "body": request.body.strip(),
            "createdAt": now,
            "updatedAt": now,
        }
        collection.insert_one(doc)
        return self._to_model(user, doc)

    def delete_comment(self, user: AuthenticatedUser, comment_id: str) -> bool:
        collection = mongo_service.get_comments_collection()
        result = collection.delete_one(
            {"id": comment_id, "auth0UserId": user.auth0_user_id}
        )
        return result.deleted_count > 0

    def _to_model(self, user: AuthenticatedUser, doc: dict) -> BookComment:
        return BookComment(
            id=doc["id"],
            book_key=doc["bookKey"],
            book_id=doc["bookId"],
            title=doc["title"],
            authors=list(doc.get("authors", [])),
            isbn=doc.get("isbn"),
            user_display_name=doc.get("userDisplayName") or "Reader",
            body=doc["body"],
            created_at=doc["createdAt"],
            updated_at=doc["updatedAt"],
            is_owner=doc.get("auth0UserId") == user.auth0_user_id,
        )

    def _book_key(
        self, book_id: str, isbn: str | None, title: str, authors: list[str]
    ) -> str:
        if isbn and isbn.strip():
            return f"isbn:{isbn.strip().lower()}"
        if book_id.startswith("/works/"):
            return f"ol:{book_id.lower()}"
        if book_id.startswith("googlebooks:"):
            return f"gb:{book_id.lower()}"

        normalized_title = self._normalize_text(title)
        normalized_author = self._normalize_text(authors[0] if authors else "")
        return f"text:{normalized_title}::{normalized_author}"

    def _normalize_text(self, value: str) -> str:
        lowered = value.strip().lower()
        return re.sub(r"\s+", " ", lowered)

    def _display_name(
        self, user: AuthenticatedUser, requested_display_name: str | None = None
    ) -> str:
        if requested_display_name and requested_display_name.strip():
            return requested_display_name.strip()
        if user.display_name and user.display_name.strip():
            return user.display_name.strip()
        if user.email and user.email.strip():
            return self._mask_email(user.email)
        return "Reader"

    def _mask_email(self, email: str) -> str:
        local, _, domain = email.partition("@")
        if not local or not domain:
            return "Reader"
        if len(local) <= 2:
            masked_local = local[0] + "*"
        else:
            masked_local = f"{local[0]}{'*' * max(1, len(local) - 2)}{local[-1]}"
        return f"{masked_local}@{domain}"


comment_service = CommentService()
