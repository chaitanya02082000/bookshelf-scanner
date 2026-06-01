import {Injectable, inject} from "@angular/core";
import {firstValueFrom} from "rxjs";
import {AuthService} from "@auth0/auth0-angular";
import {Book, BookComment, CreateBookCommentRequest} from "@/core/models";
import {environment} from "@/../environments/environment";

interface ApiResult<T> {
  success: boolean;
  data?: T;
  error?: string;
}

@Injectable({
  providedIn: "root",
})
export class CommentService {
  private readonly auth = inject(AuthService);
  private readonly apiUrl = environment.apiUrl;

  async listBookComments(book: Book, limit = 100): Promise<BookComment[]> {
    const result = await this.authorizedFetch<BookComment[]>(`${this.apiUrl}/comments/books/list?limit=${limit}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(this.bookPayload(book)),
    });

    return result?.success ? result.data ?? [] : [];
  }

  async createBookComment(book: Book, body: string): Promise<BookComment | null> {
    const payload: CreateBookCommentRequest = {
      ...this.bookPayload(book),
      body,
    };
    const result = await this.authorizedFetch<BookComment>(`${this.apiUrl}/comments/books`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    return result?.success ? result.data ?? null : null;
  }

  async deleteBookComment(commentId: string): Promise<boolean> {
    const result = await this.authorizedFetch<boolean>(`${this.apiUrl}/comments/${encodeURIComponent(commentId)}`, {
      method: "DELETE",
    });

    return !!result?.success;
  }

  private bookPayload(book: Book) {
    return {
      bookId: book.id,
      title: book.title,
      authors: book.authors,
      isbn: book.isbn ?? null,
    };
  }

  private async authorizedFetch<T>(input: string, init?: RequestInit) {
    const token = await firstValueFrom(this.auth.getAccessTokenSilently());
    const response = await fetch(input, {
      ...init,
      headers: {
        ...(init?.headers ?? {}),
        Authorization: `Bearer ${token}`,
        "ngrok-skip-browser-warning": "1",
      },
    });

    if (!response.ok) {
      return null;
    }

    return (await response.json()) as ApiResult<T>;
  }
}
