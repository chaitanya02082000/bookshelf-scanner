import {Injectable, signal} from "@angular/core";
import {Book, BookSearchResult, LibgenBook, LibgenSearchResult} from "@/core/models";
import {environment} from "@/../environments/environment";

interface OpenLibrarySearchDoc {
  key: string;
  title?: string;
  author_name?: string[];
  cover_i?: number;
  first_publish_year?: number;
  isbn?: string[];
  subject?: string[];
}

interface OpenLibrarySearchResponse {
  numFound: number;
  docs: OpenLibrarySearchDoc[];
}

interface OpenLibraryWorkResponse {
  title?: string;
  description?: string | {value?: string};
  subjects?: string[];
  covers?: number[];
}

@Injectable({
  providedIn: "root",
})
export class BookCatalogService {
  readonly isLoading = signal(false);
  private readonly apiUrl = environment.apiUrl;

  async search(query: string, limit = 12): Promise<BookSearchResult> {
    if (!query.trim()) {
      return {query, total: 0, items: []};
    }

    this.isLoading.set(true);
    try {
      const url = new URL("https://openlibrary.org/search.json");
      url.searchParams.set("q", query);
      url.searchParams.set("limit", String(limit));
      const response = await fetch(url.toString());
      const data = (await response.json()) as OpenLibrarySearchResponse;
      const items = (data.docs ?? []).map((doc) => this.mapDocToBook(doc));
      return {
        query,
        total: data.numFound ?? items.length,
        items,
      };
    } finally {
      this.isLoading.set(false);
    }
  }

  async getWorkDetails(workKey: string): Promise<Partial<Book>> {
    if (!workKey) {
      return {};
    }
    const response = await fetch(`https://openlibrary.org${workKey}.json`);
    const data = (await response.json()) as OpenLibraryWorkResponse;
    const description =
      typeof data.description === "string"
        ? data.description
        : data.description?.value;

    return {
      description: description ?? null,
      subjects: data.subjects ?? [],
      coverUrl: data.covers?.length
        ? `https://covers.openlibrary.org/b/id/${data.covers[0]}-L.jpg`
        : null,
    };
  }

  async searchLibgen(query: string, limit = 8): Promise<LibgenSearchResult> {
    if (!query.trim()) {
      return {query, total: 0, items: []};
    }

    this.isLoading.set(true);
    try {
      const url = new URL(`${this.apiUrl}/catalog/libgen/search`);
      url.searchParams.set("q", query);
      url.searchParams.set("limit", String(limit));
      const response = await fetch(url.toString(), {
        headers: {
          "ngrok-skip-browser-warning": "1",
        },
      });
      const result = (await response.json()) as {
        success: boolean;
        data?: LibgenBook[];
      };
      const items = result.data ?? [];
      return {
        query,
        total: items.length,
        items,
      };
    } finally {
      this.isLoading.set(false);
    }
  }

  async resolveLibgenDownloadLink(query: string, book: LibgenBook): Promise<string | null> {
    const response = await fetch(`${this.apiUrl}/catalog/libgen/resolve`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "1",
      },
      body: JSON.stringify({
        query,
        md5: book.md5,
        title: book.title,
        author: book.author ?? null,
      }),
    });

    if (!response.ok) {
      return null;
    }

    const result = (await response.json()) as {success: boolean; data?: string};
    return result.data ?? null;
  }

  private mapDocToBook(doc: OpenLibrarySearchDoc): Book {
    return {
      id: doc.key,
      title: doc.title ?? "Untitled",
      authors: doc.author_name ?? ["Unknown"],
      coverUrl: doc.cover_i
        ? `https://covers.openlibrary.org/b/id/${doc.cover_i}-M.jpg`
        : null,
      publishedDate: doc.first_publish_year
        ? String(doc.first_publish_year)
        : null,
      subjects: doc.subject?.slice(0, 4) ?? [],
      isbn: doc.isbn?.[0] ?? null,
      source: "openlibrary",
    };
  }
}
