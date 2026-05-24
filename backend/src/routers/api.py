from fastapi import APIRouter
from .catalog_router import catalog_router
from .library_router import library_router
from .predict_router import predict_router

api_router = APIRouter(prefix="/api")
api_router.include_router(predict_router)
api_router.include_router(catalog_router)
api_router.include_router(library_router)
