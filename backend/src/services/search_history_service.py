from __future__ import annotations

from datetime import datetime, timezone
import re

from src.models import SearchHistoryEvent, SearchHistoryEventInput
from src.services.auth_service import AuthenticatedUser
from src.services.mongo_service import mongo_service


class SearchHistoryService:
    def list_history(
        self, user: AuthenticatedUser, limit: int = 50
    ) -> list[SearchHistoryEvent]:
        collection = mongo_service.get_search_history_collection()
        docs = collection.find(
            {"auth0UserId": user.auth0_user_id},
            {"_id": 0, "auth0UserId": 0},
        ).sort("createdAt", -1).limit(limit)
        return [SearchHistoryEvent.model_validate(doc) for doc in docs]

    def record_event(
        self, user: AuthenticatedUser, event: SearchHistoryEventInput
    ) -> SearchHistoryEvent:
        collection = mongo_service.get_search_history_collection()
        now = datetime.now(timezone.utc).isoformat()
        normalized_query = self._normalize_query(event.query)
        saved = SearchHistoryEvent(
            **event.model_dump(by_alias=False),
            normalized_query=normalized_query,
            created_at=now,
        )
        payload = saved.model_dump(by_alias=True)
        collection.insert_one(
            {
                **payload,
                "auth0UserId": user.auth0_user_id,
            }
        )
        return saved

    def _normalize_query(self, query: str) -> str:
        lowered = query.strip().lower()
        return re.sub(r"\s+", " ", lowered)


search_history_service = SearchHistoryService()
