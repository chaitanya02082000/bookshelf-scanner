from __future__ import annotations

import os

from fastapi import HTTPException, status
from pymongo import MongoClient
from pymongo.collection import Collection


class MongoService:
    def __init__(self) -> None:
        self.uri = os.getenv("MONGODB_URI", "")
        self.db_name = os.getenv("MONGODB_DB", "bookshelf")
        self.books_collection_name = os.getenv("MONGODB_BOOKS_COLLECTION", "books")
        self.search_history_collection_name = os.getenv(
            "MONGODB_SEARCH_HISTORY_COLLECTION", "search_history"
        )
        self._client: MongoClient | None = None

    def _get_client(self) -> MongoClient:
        if not self.uri:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="MONGODB_URI is not configured",
            )
        if self._client is None:
            self._client = MongoClient(self.uri)
        return self._client

    def get_books_collection(self) -> Collection:
        client = self._get_client()
        db = client[self.db_name]
        return db[self.books_collection_name]

    def get_search_history_collection(self) -> Collection:
        client = self._get_client()
        db = client[self.db_name]
        return db[self.search_history_collection_name]


mongo_service = MongoService()
