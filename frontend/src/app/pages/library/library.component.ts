import {ChangeDetectionStrategy, Component, computed, effect, signal} from "@angular/core";
import {CommonModule} from "@angular/common";
import {inject} from "@angular/core";
import {Book, BookRecommendation, ExternalEbookResult, PriceOffer} from "@/core/models";
import {BookCatalogService, LibraryService, RecommendationService} from "@/core/services";
import {formatBookSummaryHtml} from "@/core/utils/book-summary";

@Component({
  selector: "app-library",
  standalone: true,
  templateUrl: "./library.component.html",
  styleUrl: "./library.component.scss",
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule],
})
export class LibraryComponent {
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
  protected readonly priceOffers = signal<PriceOffer[]>([]);
  protected readonly isLoadingPrices = signal(false);
  protected readonly hasRequestedPrices = signal(false);
  protected readonly ebookResults = signal<ExternalEbookResult[]>([]);
  protected readonly isLoadingEbooks = signal(false);
  protected readonly hasRequestedEbooks = signal(false);
  protected readonly ebookResolvingId = signal<string | null>(null);
  protected readonly ebookError = signal("");
  protected readonly modalError = signal("");

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
    this.isLoadingPrices.set(false);
    this.hasRequestedPrices.set(false);
    this.isLoadingEbooks.set(false);
    this.hasRequestedEbooks.set(false);
    this.ebookResolvingId.set(null);

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

  closeBookModal() {
    this.modalOpen.set(false);
    this.selectedBook.set(null);
    this.priceOffers.set([]);
    this.ebookResults.set([]);
    this.hasRequestedPrices.set(false);
    this.hasRequestedEbooks.set(false);
    this.modalError.set("");
    this.ebookError.set("");
    this.ebookResolvingId.set(null);
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
}
