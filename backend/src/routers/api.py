from fastapi import APIRouter
from .catalog_router import catalog_router
from .debug_router import debug_router
from .library_router import library_router
from .predict_router import predict_router
from .price_router import price_router
from .recommendation_router import recommendation_router
from .search_history_router import search_history_router

api_router = APIRouter(prefix="/api")
api_router.include_router(predict_router)
api_router.include_router(catalog_router)
api_router.include_router(library_router)
api_router.include_router(price_router)
api_router.include_router(search_history_router)
api_router.include_router(recommendation_router)
api_router.include_router(debug_router)
