from fastapi import APIRouter, Depends, HTTPException, Query

from src.models import (
    BookComment,
    BookCommentListRequest,
    CreateBookCommentRequest,
    ResultWithArray,
    ResultWithData,
)
from src.services import AuthenticatedUser, comment_service, get_current_user


comment_router = APIRouter(prefix="/comments", tags=["comments"])


@comment_router.post("/books/list")
def list_book_comments(
    request: BookCommentListRequest,
    limit: int = Query(100, ge=1, le=200),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ResultWithArray[BookComment]:
    comments = comment_service.list_comments(user, request, limit=limit)
    return ResultWithArray[BookComment].succeed(comments)


@comment_router.post("/books")
def create_book_comment(
    request: CreateBookCommentRequest,
    user: AuthenticatedUser = Depends(get_current_user),
) -> ResultWithData[BookComment]:
    saved = comment_service.create_comment(user, request)
    return ResultWithData[BookComment].succeed(saved)


@comment_router.delete("/{comment_id}")
def delete_book_comment(
    comment_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> ResultWithData[bool]:
    deleted = comment_service.delete_comment(user, comment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Comment not found")
    return ResultWithData[bool].succeed(True)
