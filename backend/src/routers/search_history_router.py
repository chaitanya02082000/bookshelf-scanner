from fastapi import APIRouter, Depends, Query

from src.models import ResultWithArray, ResultWithData, SearchHistoryEvent, SearchHistoryEventInput
from src.services import AuthenticatedUser, get_current_user
from src.services.search_history_service import search_history_service


search_history_router = APIRouter(prefix="/search", tags=["search"])


@search_history_router.get("/history")
def list_search_history(
    limit: int = Query(50, ge=1, le=100),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ResultWithArray[SearchHistoryEvent]:
    history = search_history_service.list_history(user, limit=limit)
    return ResultWithArray[SearchHistoryEvent].succeed(history)


@search_history_router.post("/history")
def record_search_history(
    event: SearchHistoryEventInput,
    user: AuthenticatedUser = Depends(get_current_user),
) -> ResultWithData[SearchHistoryEvent]:
    saved = search_history_service.record_event(user, event)
    return ResultWithData[SearchHistoryEvent].succeed(saved)
