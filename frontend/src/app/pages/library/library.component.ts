import {ChangeDetectionStrategy, Component, computed, signal} from "@angular/core";
import {CommonModule} from "@angular/common";
import {inject} from "@angular/core";
import {Book, PriceOffer} from "@/core/models";
import {BookCatalogService, LibraryService} from "@/core/services";
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
  private readonly libraryService = inject(LibraryService);
  private readonly catalog = inject(BookCatalogService);
  protected readonly books = this.libraryService.books;
  protected readonly hasBooks = computed(() => this.books().length > 0);
  protected readonly selectedBook = signal<Book | null>(null);
  protected readonly modalOpen = signal(false);
  protected readonly priceOffers = signal<PriceOffer[]>([]);
  protected readonly isLoadingPrices = signal(false);
  protected readonly hasRequestedPrices = signal(false);
  protected readonly modalError = signal("");

  async removeBook(id: string) {
    await this.libraryService.removeBook(id);
  }

  async openBookModal(book: Book) {
    this.selectedBook.set(book);
    this.modalOpen.set(true);
    this.priceOffers.set([]);
    this.modalError.set("");
    this.isLoadingPrices.set(false);
    this.hasRequestedPrices.set(false);

    const summaryText = this.getMeaningfulText(book.summary) || this.getMeaningfulText(book.description);
    if (summaryText) {
      return;
    }

    const enrichedBook = await this.libraryService.ensureBookSummary(book);
    if (this.selectedBook()?.id === enrichedBook.id) {
      this.selectedBook.set(enrichedBook);
    }
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

  closeBookModal() {
    this.modalOpen.set(false);
    this.selectedBook.set(null);
    this.priceOffers.set([]);
    this.hasRequestedPrices.set(false);
    this.modalError.set("");
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

  private getMeaningfulText(value?: string | null): string | null {
    const text = value?.trim();
    if (!text || text.startsWith("Imported from scan:")) {
      return null;
    }
    return text;
  }
}
