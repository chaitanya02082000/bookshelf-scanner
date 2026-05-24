import {Injectable, inject, signal} from "@angular/core";
import {firstValueFrom, Subscription} from "rxjs";
import {AuthService} from "@auth0/auth0-angular";
import {Book} from "@/core/models";
import {BookCatalogService} from "@/core/services/book-catalog.service";
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
  private readonly catalog = inject(BookCatalogService);
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

    const enrichedBook = await this.enrichBook(book);
    const result = await this.saveBook(enrichedBook);

    if (result?.success && result.data) {
      this.libraryState.update((state) => [result.data!, ...state]);
    }
  }

  async ensureBookSummary(book: Book): Promise<Book> {
    const enrichedBook = await this.enrichBook(book);
    if (enrichedBook === book) {
      return book;
    }

    const result = await this.saveBook(enrichedBook);
    if (result?.success && result.data) {
      this.libraryState.update((state) =>
        state.map((item) => (item.id === result.data!.id ? result.data! : item))
      );
      return result.data;
    }

    return enrichedBook;
  }

  private async saveBook(book: Book) {
    const payload = {...book, source: book.source ?? "scan"};
    return this.authorizedFetch<Book>(`${this.apiUrl}/library/books`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  }

  private async enrichBook(book: Book): Promise<Book> {
    const supportsRemoteDetails =
      book.id.startsWith("/works/") || book.id.startsWith("googlebooks:");
    const hasSummary =
      this.hasMeaningfulText(book.summary) || this.hasMeaningfulText(book.description);
    if (!supportsRemoteDetails || hasSummary) {
      return book;
    }

    try {
      const details = await this.catalog.getWorkDetails(book.id, this.buildFallbackQuery(book));
      const summary = details.summary ?? details.description ?? null;
      const currentDescription = this.hasMeaningfulText(book.description)
        ? book.description
        : null;
      return {
        ...book,
        ...details,
        summary,
        description: details.description ?? summary ?? currentDescription,
      };
    } catch {
      return book;
    }
  }

  private hasMeaningfulText(value?: string | null) {
    const text = value?.trim();
    return !!text && !text.startsWith("Imported from scan:");
  }

  private buildFallbackQuery(book: Book) {
    const author = book.authors[0]?.trim();
    return author ? `${book.title} ${author}` : book.title;
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
        "ngrok-skip-browser-warning": "1",
      },
    });

    if (!response.ok) {
      return null;
    }

    return (await response.json()) as ApiResult<T>;
  }
}
