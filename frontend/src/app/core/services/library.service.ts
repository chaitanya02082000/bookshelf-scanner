import {Injectable, inject, signal} from "@angular/core";
import {firstValueFrom, Subscription} from "rxjs";
import {AuthService} from "@auth0/auth0-angular";
import {Book} from "@/core/models";
import {environment} from "@/../environments/environment";

interface ApiResult<T> {
  success: boolean;
  data?: T;
  error?: string;
}

@Injectable({
  providedIn: "root",
})
export class LibraryService {
  private readonly auth = inject(AuthService);
  private readonly apiUrl = environment.apiUrl;
  private readonly libraryState = signal<Book[]>([]);
  private authSubscription: Subscription;
  readonly books = this.libraryState.asReadonly();

  constructor() {
    this.authSubscription = this.auth.isAuthenticated$.subscribe((isAuthenticated) => {
      if (isAuthenticated) {
        void this.loadRemote();
        return;
      }
      this.libraryState.set([]);
    });
  }

  async addBook(book: Book) {
    const existing = this.libraryState().find((item) => item.id === book.id);
    if (existing) {
      return;
    }

    const payload = {...book, source: book.source ?? "scan"};
    const result = await this.authorizedFetch<Book>(`${this.apiUrl}/library/books`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (result?.success && result.data) {
      this.libraryState.update((state) => [result.data!, ...state]);
    }
  }

  async removeBook(id: string) {
    const result = await this.authorizedFetch<boolean>(`${this.apiUrl}/library/books/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });

    if (result?.success) {
      this.libraryState.update((state) => state.filter((book) => book.id !== id));
    }
  }

  clear() {
    this.libraryState.set([]);
  }

  hydrate(books: Book[]) {
    this.libraryState.set(books);
  }

  private async loadRemote() {
    const result = await this.authorizedFetch<Book[]>(`${this.apiUrl}/library/books`);
    if (result?.success) {
      this.libraryState.set(result.data ?? []);
    }
  }

  private async authorizedFetch<T>(input: string, init?: RequestInit) {
    const token = await firstValueFrom(this.auth.getAccessTokenSilently());
    const response = await fetch(input, {
      ...init,
      headers: {
        ...(init?.headers ?? {}),
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      return null;
    }

    return (await response.json()) as ApiResult<T>;
  }
}
