import { Injectable, inject, signal } from "@angular/core";
import { firstValueFrom } from "rxjs";
import { AuthService } from "@auth0/auth0-angular";
import {
    Book,
    BookSearchResult,
    ExternalEbookResult,
    ExternalEbookSearchResult,
    LibgenBook,
    LibgenSearchResult,
    PriceOffer,
} from "@/core/models";
import { environment } from "@/../environments/environment";

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
    description?: string | { value?: string };
    subjects?: string[];
    covers?: number[];
}

interface GoogleBooksVolumeInfo {
    title?: string;
    authors?: string[];
    description?: string;
    publishedDate?: string;
    pageCount?: number;
    industryIdentifiers?: Array<{ type?: string; identifier?: string }>;
    categories?: string[];
    imageLinks?: {
        thumbnail?: string;
        smallThumbnail?: string;
    };
}

interface GoogleBooksVolume {
    id: string;
    volumeInfo?: GoogleBooksVolumeInfo;
}

interface GoogleBooksResponse {
    totalItems?: number;
    items?: GoogleBooksVolume[];
}

interface ExternalEbookApiResult {
    title?: string;
    author?: string;
    publisher?: string;
    year?: string;
    language?: string;
    pages?: string | number;
    filesize?: string | number;
    format?: string;
    source?: string;
    coverUrl?: string;
    downloadUrl?: string;
    directUrl?: string;
    directReady?: boolean;
    md5?: string;
    _score?: number;
}

interface ExternalEbookApiResponse {
    source?: string;
    results?: ExternalEbookApiResult[];
}

@Injectable({
    providedIn: "root",
})
export class BookCatalogService {
    readonly isLoading = signal(false);
    private readonly auth = inject(AuthService);
    private readonly apiUrl = environment.apiUrl;
    private readonly googleBooksApiKey = environment.googleBooksApiKey;
    // #EDIT Replace with your approved ebook provider endpoint.
    private readonly externalEbookSearchUrl = "https://ravebooksearch.cloudflare-s3cvv.workers.dev/search/all";

    async search(query: string, limit = 12): Promise<BookSearchResult> {
        if (!query.trim()) {
            return { query, total: 0, items: [] };
        }

        this.isLoading.set(true);
        try {
            const [openLibraryData, googleBooksData] = await Promise.all([
                this.searchOpenLibrary(query, limit),
                this.searchGoogleBooks(query, limit),
            ]);
            const openLibraryItems = (openLibraryData.docs ?? []).map((doc) => this.mapDocToBook(doc));
            const googleBooksItems = (googleBooksData.items ?? []).map((item) => this.mapGoogleBookToBook(item));
            const items = this.mergeBooks(openLibraryItems, googleBooksItems).slice(0, limit);

            return {
                query,
                total: items.length,
                items,
            };
        } finally {
            this.isLoading.set(false);
        }
    }

    async searchLowConfidenceCandidates(query: string): Promise<Book[]> {
        if (!query.trim()) {
            return [];
        }

        const [openLibraryData, googleBooksData] = await Promise.all([
            this.searchOpenLibrary(query, 12),
            this.searchGoogleBooks(query, 12),
        ]);
        const openLibraryItems = (openLibraryData.docs ?? []).map((doc) => this.mapDocToBook(doc));
        const googleBooksItems = (googleBooksData.items ?? []).map((item) => this.mapGoogleBookToBook(item));
        return this.mergeBooks(openLibraryItems, googleBooksItems);
    }

    async getWorkDetails(workKey: string, fallbackQuery?: string): Promise<Partial<Book>> {
        if (!workKey) {
            return {};
        }

        if (workKey.startsWith("googlebooks:")) {
            return this.getGoogleBookDetails(workKey.replace("googlebooks:", ""));
        }

        const response = await fetch(`https://openlibrary.org${workKey}.json`);
        const data = (await response.json()) as OpenLibraryWorkResponse;
        const description =
            typeof data.description === "string"
                ? data.description
                : data.description?.value;

        const baseDetails: Partial<Book> = {
            description: description ?? null,
            summary: description ?? null,
            subjects: data.subjects ?? [],
            coverUrl: data.covers?.length
                ? `https://covers.openlibrary.org/b/id/${data.covers[0]}-L.jpg`
                : null,
        };

        if (description?.trim()) {
            return baseDetails;
        }

        const fallback = await this.lookupGoogleBooksSummary(
            fallbackQuery ?? data.title ?? ""
        );
        return {
            ...baseDetails,
            ...fallback,
            description: fallback.description ?? baseDetails.description ?? null,
            summary: fallback.summary ?? baseDetails.summary ?? null,
        };
    }

    async searchLibgen(query: string, limit = 8): Promise<LibgenSearchResult> {
        if (!query.trim()) {
            return { query, total: 0, items: [] };
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

        const result = (await response.json()) as { success: boolean; data?: string };
        return result.data ?? null;
    }

    async searchPrices(query: string, limit = 3): Promise<PriceOffer[]> {
        if (!query.trim()) {
            return [];
        }

        const token = await firstValueFrom(this.auth.getAccessTokenSilently());
        const url = new URL(`${this.apiUrl}/pricing/search`);
        url.searchParams.set("q", query);
        url.searchParams.set("limit", String(limit));

        const response = await fetch(url.toString(), {
            headers: {
                Authorization: `Bearer ${token}`,
                "ngrok-skip-browser-warning": "1",
            },
        });

        if (!response.ok) {
            return [];
        }

        const result = (await response.json()) as {
            success: boolean;
            data?: PriceOffer[];
        };
        return result.data ?? [];
    }

    async searchExternalEbooks(query: string, limit = 12): Promise<ExternalEbookSearchResult> {
        if (!query.trim()) {
            return { query, total: 0, items: [] };
        }

        // #EDIT Adjust query params if your provider expects a different contract.
        const url = new URL(this.externalEbookSearchUrl);
        url.searchParams.set("q", query);
        url.searchParams.set("mode", "ebooks");

        const response = await fetch(url.toString());
        if (!response.ok) {
            throw new Error(`External ebook search failed with status ${response.status}`);
        }

        const payload = (await response.json()) as ExternalEbookApiResponse;
        const items = (payload.results ?? [])
            .map((item, index) => this.mapExternalEbookResult(item, index))
            .sort((a, b) => (b.score ?? Number.NEGATIVE_INFINITY) - (a.score ?? Number.NEGATIVE_INFINITY))
            .slice(0, limit);

        return {
            query,
            total: items.length,
            items,
        };
    }

    async resolveExternalEbookDownload(book: ExternalEbookResult): Promise<string | null> {
        // #EDIT Replace this with a follow-up resolve call if your provider requires one.
        return book.resolvedDownloadLink ?? book.downloadUrl ?? null;
    }

    private async searchOpenLibrary(query: string, limit: number): Promise<OpenLibrarySearchResponse> {
        const url = new URL("https://openlibrary.org/search.json");
        url.searchParams.set("q", query);
        url.searchParams.set("limit", String(limit));
        const response = await fetch(url.toString());
        return (await response.json()) as OpenLibrarySearchResponse;
    }

    private async searchGoogleBooks(query: string, limit: number): Promise<GoogleBooksResponse> {
        const url = this.createGoogleBooksUrl("volumes");
        url.searchParams.set("q", query);
        url.searchParams.set("maxResults", String(Math.min(limit, 20)));
        url.searchParams.set("langRestrict", "en");
        const response = await fetch(url.toString());
        return (await response.json()) as GoogleBooksResponse;
    }

    private async getGoogleBookDetails(volumeId: string): Promise<Partial<Book>> {
        if (!volumeId) {
            return {};
        }

        const response = await fetch(this.createGoogleBooksUrl(`volumes/${volumeId}`).toString());
        const data = (await response.json()) as GoogleBooksVolume;
        const mapped = this.mapGoogleBookToBook(data);
        return {
            description: mapped.description ?? null,
            summary: mapped.summary ?? null,
            subjects: mapped.subjects ?? [],
            coverUrl: mapped.coverUrl ?? null,
            publishedDate: mapped.publishedDate ?? null,
            pageCount: mapped.pageCount ?? null,
            isbn: mapped.isbn ?? null,
        };
    }

    private async lookupGoogleBooksSummary(query: string): Promise<Partial<Book>> {
        if (!query.trim()) {
            return {};
        }

        const data = await this.searchGoogleBooks(query, 1);
        const first = data.items?.[0];
        if (!first) {
            return {};
        }

        const mapped = this.mapGoogleBookToBook(first);
        return {
            description: mapped.description ?? null,
            summary: mapped.summary ?? null,
            subjects: mapped.subjects ?? [],
            coverUrl: mapped.coverUrl ?? null,
            publishedDate: mapped.publishedDate ?? null,
            pageCount: mapped.pageCount ?? null,
            isbn: mapped.isbn ?? null,
        };
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

    private mapGoogleBookToBook(volume: GoogleBooksVolume): Book {
        const info = volume.volumeInfo ?? {};
        const description = info.description ?? null;
        const isbn =
            info.industryIdentifiers?.find((identifier) => identifier.type?.includes("ISBN"))?.identifier ?? null;

        return {
            id: `googlebooks:${volume.id}`,
            title: info.title ?? "Untitled",
            authors: info.authors?.length ? info.authors : ["Unknown"],
            coverUrl: info.imageLinks?.thumbnail ?? info.imageLinks?.smallThumbnail ?? null,
            description,
            summary: description,
            subjects: info.categories ?? [],
            publishedDate: info.publishedDate ?? null,
            pageCount: info.pageCount ?? null,
            isbn,
            source: "googlebooks",
        };
    }

    private mergeBooks(primary: Book[], secondary: Book[]): Book[] {
        const seen = new Set<string>();
        const items: Book[] = [];

        for (const book of [...primary, ...secondary]) {
            const key = `${this.normalizeText(book.title)}::${this.normalizeText(book.authors[0] ?? "")}`;
            if (seen.has(key)) {
                continue;
            }
            seen.add(key);
            items.push(book);
        }

        return items;
    }

    private normalizeText(value: string): string {
        return value.trim().toLowerCase().replace(/\s+/g, " ");
    }

    private mapExternalEbookResult(item: ExternalEbookApiResult, index: number): ExternalEbookResult {
        const title = item.title?.trim() || "Untitled";
        const source = item.source?.trim() || "External source";

        return {
            id: item.md5?.trim() || `${source}-${title}-${index}`,
            title,
            author: item.author?.trim() || null,
            publisher: item.publisher?.trim() || null,
            year: item.year?.trim() || null,
            language: item.language?.trim() || null,
            pages: item.pages != null ? String(item.pages) : null,
            size: item.filesize != null ? String(item.filesize) : null,
            extension: item.format?.trim() || null,
            source,
            score: typeof item._score === "number" ? item._score : null,
            coverUrl: item.coverUrl?.trim() || null,
            downloadUrl: item.downloadUrl?.trim() || null,
            resolvedDownloadLink:
                item.directReady && item.directUrl?.trim()
                    ? item.directUrl.trim()
                    : item.downloadUrl?.trim() || null,
        };
    }

    private createGoogleBooksUrl(path: string): URL {
        const url = new URL(`https://www.googleapis.com/books/v1/${path}`);
        url.searchParams.set("key", this.googleBooksApiKey);
        return url;
    }
}
