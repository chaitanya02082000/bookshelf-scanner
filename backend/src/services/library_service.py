from __future__ import annotations

from datetime import datetime, timezone

from src.models import BookMetadata
from src.services.auth_service import AuthenticatedUser
from src.services.mongo_service import mongo_service


class LibraryService:
    def list_books(self, user: AuthenticatedUser) -> list[BookMetadata]:
        collection = mongo_service.get_books_collection()
        docs = collection.find(
            {"auth0UserId": user.auth0_user_id},
            {"_id": 0, "auth0UserId": 0, "createdAt": 0, "updatedAt": 0},
        )
        return [BookMetadata.model_validate(doc) for doc in docs]

    def upsert_book(self, user: AuthenticatedUser, book: BookMetadata) -> BookMetadata:
        collection = mongo_service.get_books_collection()
        now = datetime.now(timezone.utc).isoformat()
        payload = book.model_dump(by_alias=False)
        collection.update_one(
            {"auth0UserId": user.auth0_user_id, "id": book.id},
            {
                "$set": {
                    **payload,
                    "auth0UserId": user.auth0_user_id,
                    "updatedAt": now,
                },
                "$setOnInsert": {"createdAt": now},
            },
            upsert=True,
        )
        return book

    def delete_book(self, user: AuthenticatedUser, book_id: str) -> bool:
        collection = mongo_service.get_books_collection()
        result = collection.delete_one(
            {"auth0UserId": user.auth0_user_id, "id": book_id}
        )
        return result.deleted_count > 0


library_service = LibraryService()
