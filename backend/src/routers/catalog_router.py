from fastapi import APIRouter, HTTPException, Query

from src.models import (
    LibgenBook,
    LibgenResolveRequest,
    ResultWithArray,
    ResultWithData,
)
from src.services import LibgenService


catalog_router = APIRouter(prefix="/catalog", tags=["catalog"])

libgen_service = LibgenService()


@catalog_router.get("/libgen/search")
def search_libgen(
    q: str = Query(..., min_length=1),
    limit: int = Query(8, ge=1, le=24),
) -> ResultWithArray[LibgenBook]:
    results = libgen_service.search(q, limit=limit)
    return ResultWithArray[LibgenBook].succeed(results)


@catalog_router.post("/libgen/resolve")
def resolve_libgen_download(request: LibgenResolveRequest) -> ResultWithData[str]:
    resolved_link = libgen_service.resolve_download_link(
        query=request.query,
        title=request.title,
        author=request.author,
        md5=request.md5,
    )
    if not resolved_link:
        raise HTTPException(
            status_code=404, detail="Download link could not be resolved"
        )
    return ResultWithData[str].succeed(resolved_link)

