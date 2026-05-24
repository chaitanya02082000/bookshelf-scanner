from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException

from src.models import BookMetadata, ResultWithArray, ResultWithData
from src.services import AuthenticatedUser, get_current_user, library_service


library_router = APIRouter(prefix="/library", tags=["library"])


@library_router.get("/books")
def list_books(
    user: AuthenticatedUser = Depends(get_current_user),
) -> ResultWithArray[BookMetadata]:
    books = library_service.list_books(user)
    return ResultWithArray[BookMetadata].succeed(books)


@library_router.put("/books")
def upsert_book(
    book: BookMetadata,
    user: AuthenticatedUser = Depends(get_current_user),
) -> ResultWithData[BookMetadata]:
    saved = library_service.upsert_book(user, book)
    return ResultWithData[BookMetadata].succeed(saved)


@library_router.delete("/books/{book_id:path}")
def delete_book(
    book_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
) -> ResultWithData[bool]:
    deleted = library_service.delete_book(user, unquote(book_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Book not found")
    return ResultWithData[bool].succeed(True)
