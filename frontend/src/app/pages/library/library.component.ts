import {ChangeDetectionStrategy, Component, computed, effect, signal} from "@angular/core";
import {CommonModule} from "@angular/common";
import {inject} from "@angular/core";
import {Book, BookRecommendation, ExternalEbookResult, PriceOffer} from "@/core/models";
import {BookCatalogService, LibraryService, RecommendationService} from "@/core/services";
import {formatBookSummaryHtml} from "@/core/utils/book-summary";

declare global {
  interface Window {
    __gcse?: {
      parsetags?: string;
      callback?: () => void;
    };
    google?: {
      search?: {
        cse?: {
          element?: {
            render: (options: {div: string; tag: string; gname: string}) => void;
            getElement: (name: string) => {execute: (query: string) => void} | null;
          };
        };
      };
    };
  }
}

let googleCseScriptPromise: Promise<void> | null = null;

interface PdfSearchResultItem {
  id: string;
  title: string;
  snippet: string | null;
  url: string;
  visibleUrl: string | null;
  fileFormat: string | null;
  sourceLabel: string | null;
  thumbnailUrl: string | null;
}

type BookModalTab = "ebooks" | "prices" | "pdf";

@Component({
  selector: "app-library",
  standalone: true,
  templateUrl: "./library.component.html",
  styleUrl: "./library.component.scss",
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule],
})
export class LibraryComponent {
  private readonly pdfCx = "006516753008110874046:s9ddesylrm8";
  private readonly pdfProbeContainerId = "library-pdf-probe";
  private readonly loaderVariants = [
    "library-loader--10",
    "library-loader--11",
    "library-loader--12",
    "library-loader--14",
    "library-loader--18",
    "library-loader--19",
  ] as const;
  private readonly libraryService = inject(LibraryService);
  private readonly catalog = inject(BookCatalogService);
  private readonly recommendationService = inject(RecommendationService);
  protected readonly books = this.libraryService.books;
  protected readonly hasBooks = computed(() => this.books().length > 0);
  protected readonly savedBookCount = computed(() => this.books().length);
  protected readonly selectedBook = signal<Book | null>(null);
  protected readonly modalOpen = signal(false);
  protected readonly recommendations = signal<BookRecommendation[]>([]);
  protected readonly isLoadingRecommendations = signal(false);
  protected readonly recommendationsError = signal("");
  protected readonly recommendationLoaderVariant = signal<string>(this.randomLoaderVariant());
  protected readonly priceLoaderVariant = signal<string>(this.randomLoaderVariant());
  protected readonly ebookLoaderVariant = signal<string>(this.randomLoaderVariant());
  protected readonly pdfLoaderVariant = signal<string>(this.randomLoaderVariant());
  protected readonly priceOffers = signal<PriceOffer[]>([]);
  protected readonly isLoadingPrices = signal(false);
  protected readonly hasRequestedPrices = signal(false);
  protected readonly ebookResults = signal<ExternalEbookResult[]>([]);
  protected readonly isLoadingEbooks = signal(false);
  protected readonly hasRequestedEbooks = signal(false);
  protected readonly ebookResolvingId = signal<string | null>(null);
  protected readonly ebookError = signal("");
  protected readonly isLoadingPdf = signal(false);
  protected readonly hasRequestedPdf = signal(false);
  protected readonly pdfError = signal("");
  protected readonly pdfResults = signal<PdfSearchResultItem[]>([]);
  protected readonly modalError = signal("");
  protected readonly activeModalTab = signal<BookModalTab>("ebooks");

  constructor() {
    effect(() => {
      this.books();
      void this.loadRecommendations();
    });
  }

  async removeBook(id: string) {
    await this.libraryService.removeBook(id);
  }

  async openBookModal(book: Book) {
    this.selectedBook.set(book);
    this.modalOpen.set(true);
    this.priceOffers.set([]);
    this.ebookResults.set([]);
    this.modalError.set("");
    this.ebookError.set("");
    this.pdfError.set("");
    this.isLoadingPrices.set(false);
    this.hasRequestedPrices.set(false);
    this.isLoadingEbooks.set(false);
    this.hasRequestedEbooks.set(false);
    this.isLoadingPdf.set(false);
    this.hasRequestedPdf.set(false);
    this.pdfResults.set([]);
    this.ebookResolvingId.set(null);
    this.activeModalTab.set("ebooks");

    const summaryText = this.getMeaningfulText(book.summary) || this.getMeaningfulText(book.description);
    if (summaryText) {
      return;
    }

    const enrichedBook = this.isBookInLibrary(book)
      ? await this.libraryService.ensureBookSummary(book)
      : await this.loadRemoteBookSummary(book);
    if (this.selectedBook()?.id === enrichedBook.id) {
      this.selectedBook.set(enrichedBook);
    }
  }

  async addRecommendedBook(book: BookRecommendation, event?: Event) {
    event?.stopPropagation();
    await this.libraryService.addBook(book);
    this.recommendations.update((items) => items.filter((item) => item.id !== book.id));
  }

  async loadPriceOffers() {
    const book = this.selectedBook();
    if (!book || this.isLoadingPrices()) {
      return;
    }

    this.hasRequestedPrices.set(true);
    this.priceOffers.set([]);
    this.modalError.set("");
    this.priceLoaderVariant.set(this.randomLoaderVariant());
    this.isLoadingPrices.set(true);

    try {
      const author = book.authors[0]?.trim();
      const query = author ? `${book.title} ${author}` : book.title;
      const offers = await this.catalog.searchPrices(query, 4);
      this.priceOffers.set(offers);
      if (!offers.length) {
        this.modalError.set("No online prices found right now.");
      } else if (offers.every((offer) => !offer.price)) {
        this.modalError.set(offers[0]?.note ?? "Open one of the online stores to compare prices.");
      }
    } catch (error) {
      this.modalError.set(`Unable to load price details: ${error}`);
    } finally {
      this.isLoadingPrices.set(false);
    }
  }

  async loadEbookOptions() {
    const book = this.selectedBook();
    if (!book || this.isLoadingEbooks()) {
      return;
    }

    this.hasRequestedEbooks.set(true);
    this.ebookResults.set([]);
    this.ebookError.set("");
    this.ebookLoaderVariant.set(this.randomLoaderVariant());
    this.isLoadingEbooks.set(true);

    try {
      const author = book.authors[0]?.trim();
      const query = author ? `${book.title} ${author}` : book.title;
      const result = await this.catalog.searchExternalEbooks(query, 8);
      this.ebookResults.set(result.items);
      if (!result.items.length) {
        this.ebookError.set("No digital editions found.");
      }
    } catch (error) {
      this.ebookError.set(`Failed to search external provider: ${error}`);
    } finally {
      this.isLoadingEbooks.set(false);
    }
  }

  async openEbookLink(book: ExternalEbookResult) {
    this.ebookResolvingId.set(book.id);
    this.ebookError.set("");
    try {
      const url = await this.catalog.resolveExternalEbookDownload(book);
      if (!url) {
        this.ebookError.set("Open link could not be resolved.");
        return;
      }
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (error) {
      this.ebookError.set(`Failed to resolve external link: ${error}`);
    } finally {
      this.ebookResolvingId.set(null);
    }
  }

  async loadPdfResults() {
    const book = this.selectedBook();
    if (!book || this.isLoadingPdf()) {
      return;
    }

    this.hasRequestedPdf.set(true);
    this.pdfError.set("");
    this.pdfResults.set([]);
    this.pdfLoaderVariant.set(this.randomLoaderVariant());
    this.isLoadingPdf.set(true);

    try {
      const results = await this.renderHiddenPdfSearch(book);
      this.pdfResults.set(results);
      if (!results.length) {
        this.pdfError.set("No PDF-style document results found.");
      }
    } catch (error) {
      this.pdfError.set(`Failed to search PDF results: ${error}`);
    } finally {
      this.isLoadingPdf.set(false);
    }
  }

  openPdfLink(item: PdfSearchResultItem) {
    window.open(item.url, "_blank", "noopener,noreferrer");
  }

  async setModalTab(tab: BookModalTab) {
    this.activeModalTab.set(tab);

    if (tab === "ebooks" && !this.hasRequestedEbooks()) {
      await this.loadEbookOptions();
    }

    if (tab === "prices" && !this.hasRequestedPrices()) {
      await this.loadPriceOffers();
    }

    if (tab === "pdf" && !this.hasRequestedPdf()) {
      await this.loadPdfResults();
    }
  }

  closeBookModal() {
    this.modalOpen.set(false);
    this.selectedBook.set(null);
    this.priceOffers.set([]);
    this.ebookResults.set([]);
    this.hasRequestedPrices.set(false);
    this.hasRequestedEbooks.set(false);
    this.hasRequestedPdf.set(false);
    this.isLoadingPrices.set(false);
    this.isLoadingEbooks.set(false);
    this.isLoadingPdf.set(false);
    this.modalError.set("");
    this.ebookError.set("");
    this.pdfError.set("");
    this.pdfResults.set([]);
    this.ebookResolvingId.set(null);
    this.activeModalTab.set("ebooks");
  }

  protected formatOfferPrice(offer: PriceOffer): string {
    if (offer.price?.trim()) {
      return offer.price;
    }
    if (offer.currency?.trim()) {
      return offer.currency;
    }
    return "Open listing";
  }

  protected formatSummary(summary?: string | null): string {
    return formatBookSummaryHtml(summary);
  }

  private async loadRecommendations() {
    this.recommendationLoaderVariant.set(this.randomLoaderVariant());
    this.isLoadingRecommendations.set(true);
    this.recommendationsError.set("");
    try {
      const recommendations = await this.recommendationService.listBooks(8);
      this.recommendations.set(recommendations);
    } catch (error) {
      this.recommendations.set([]);
      this.recommendationsError.set(`Unable to load recommendations: ${error}`);
    } finally {
      this.isLoadingRecommendations.set(false);
    }
  }

  private isBookInLibrary(book: Book) {
    return this.books().some((entry) => entry.id === book.id);
  }

  private async loadRemoteBookSummary(book: Book): Promise<Book> {
    const author = book.authors[0]?.trim();
    const fallbackQuery = author ? `${book.title} ${author}` : book.title;
    const details = await this.catalog.getWorkDetails(book.id, fallbackQuery);
    const summary = details.summary ?? details.description ?? book.summary ?? book.description ?? null;
    return {
      ...book,
      ...details,
      summary,
      description: details.description ?? summary,
    };
  }

  private getMeaningfulText(value?: string | null): string | null {
    const text = value?.trim();
    if (!text || text.startsWith("Imported from scan:")) {
      return null;
    }
    return text;
  }

  private randomLoaderVariant() {
    const index = Math.floor(Math.random() * this.loaderVariants.length);
    return this.loaderVariants[index];
  }

  private buildPdfSearchQuery(book: Book) {
    const author = book.authors[0]?.trim();
    return author ? `${book.title} ${author}` : book.title;
  }

  private async renderHiddenPdfSearch(book: Book): Promise<PdfSearchResultItem[]> {
    const query = this.buildPdfSearchQuery(book);
    const probe = await this.waitForPdfProbeContainer();
    probe.innerHTML = "";
    await this.ensureGoogleCseLoaded();

    const cse = window.google?.search?.cse?.element;
    if (!cse) {
      throw new Error("Google CSE failed to initialize.");
    }

    const gname = `libraryPdfSearch${Date.now()}`;
    cse.render({
      div: this.pdfProbeContainerId,
      tag: "searchresults-only",
      gname,
    });

    const element = cse.getElement(gname);
    if (!element) {
      throw new Error("Google CSE search element was not created.");
    }

    const resultsPromise = this.waitForRenderedPdfResults(probe);
    element.execute(query);
    return resultsPromise;
  }

  private waitForPdfProbeContainer() {
    return new Promise<HTMLElement>((resolve, reject) => {
      let attempts = 0;
      const maxAttempts = 24;

      const tryResolve = () => {
        const container = document.getElementById(this.pdfProbeContainerId);
        if (container) {
          resolve(container);
          return;
        }

        attempts += 1;
        if (attempts >= maxAttempts) {
          reject(new Error("PDF probe container was not found."));
          return;
        }

        requestAnimationFrame(tryResolve);
      };

      tryResolve();
    });
  }

  private ensureGoogleCseLoaded() {
    if (window.google?.search?.cse?.element) {
      return Promise.resolve();
    }

    if (googleCseScriptPromise) {
      return googleCseScriptPromise;
    }

    googleCseScriptPromise = new Promise<void>((resolve, reject) => {
      window.__gcse = {
        parsetags: "explicit",
        callback: () => resolve(),
      };

      const existingScript = document.querySelector<HTMLScriptElement>(`script[src*="cse.google.com/cse.js?cx=${this.pdfCx}"]`);
      if (existingScript) {
        existingScript.addEventListener("load", () => resolve(), {once: true});
        existingScript.addEventListener("error", () => reject(new Error("Google CSE script failed to load.")), {
          once: true,
        });
        return;
      }

      const script = document.createElement("script");
      script.type = "text/javascript";
      script.async = true;
      script.defer = true;
      script.src = `https://cse.google.com/cse.js?cx=${this.pdfCx}`;
      script.onerror = () => reject(new Error("Google CSE script failed to load."));
      document.head.appendChild(script);
    });

    return googleCseScriptPromise;
  }

  private waitForRenderedPdfResults(container: HTMLElement) {
    return new Promise<PdfSearchResultItem[]>((resolve) => {
      let settled = false;

      const finish = () => {
        if (settled) {
          return;
        }
        settled = true;
        observer.disconnect();
        resolve(this.extractPdfResults(container));
      };

      const observer = new MutationObserver(() => {
        const hasResults = container.querySelector(".gsc-webResult.gsc-result, .gs-webResult.gs-result");
        const noResults = container.querySelector(".gs-no-results-result");
        if (!hasResults && !noResults) {
          return;
        }
        finish();
      });

      observer.observe(container, {childList: true, subtree: true});
      window.setTimeout(finish, 12000);
    });
  }

  private extractPdfResults(container: HTMLElement): PdfSearchResultItem[] {
    const nodes = Array.from(
      container.querySelectorAll<HTMLElement>(".gsc-webResult.gsc-result, .gs-webResult.gs-result")
    );

    return nodes
      .map((node, index) => {
        const link = node.querySelector<HTMLAnchorElement>("a.gs-title, .gs-title a");
        const title = link?.textContent?.trim();
        const url = link?.href?.trim();

        if (!title || !url) {
          return null;
        }

        const snippet = node.querySelector<HTMLElement>(".gs-snippet")?.textContent?.trim() || null;
        const visibleUrl = node.querySelector<HTMLElement>(".gs-visibleUrl")?.textContent?.trim() || null;
        const fileFormat = node.querySelector<HTMLElement>(".gs-fileFormat")?.textContent?.trim() || null;
        const sourceLabel = node.querySelector<HTMLElement>(".gs-visibleUrl")?.textContent?.trim() || null;
        const thumbnailUrl = node.querySelector<HTMLImageElement>("img")?.src || null;

        return {
          id: `pdf-${index}-${url}`,
          title,
          snippet,
          url,
          visibleUrl,
          fileFormat,
          sourceLabel,
          thumbnailUrl,
        } satisfies PdfSearchResultItem;
      })
      .filter((item): item is PdfSearchResultItem => item !== null);
  }
}
