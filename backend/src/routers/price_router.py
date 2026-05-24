from fastapi import APIRouter, Depends, Query

from src.models import PriceOffer, ResultWithArray
from src.services import AuthenticatedUser, get_current_user, price_service


price_router = APIRouter(prefix="/pricing", tags=["pricing"])


@price_router.get("/search")
def search_prices(
    q: str = Query(..., min_length=1),
    limit: int = Query(3, ge=1, le=6),
    _: AuthenticatedUser = Depends(get_current_user),
) -> ResultWithArray[PriceOffer]:
    offers = price_service.search_book_prices(q, limit=limit)
    return ResultWithArray[PriceOffer].succeed(offers)
