from fastapi import APIRouter, Depends, Query

from src.models import BookRecommendation, ResultWithArray
from src.services import AuthenticatedUser, get_current_user
from src.services.recommendation_service import recommendation_service


recommendation_router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@recommendation_router.get("/books")
def list_book_recommendations(
    limit: int = Query(12, ge=1, le=24),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ResultWithArray[BookRecommendation]:
    recommendations = recommendation_service.list_recommendations(user, limit=limit)
    return ResultWithArray[BookRecommendation].succeed(recommendations)
