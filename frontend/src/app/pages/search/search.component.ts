import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  viewChild,
  signal,
} from "@angular/core";
import {CommonModule} from "@angular/common";
import {FormsModule} from "@angular/forms";
import {BookCatalogService, LibraryService} from "@/core/services";
import {Book} from "@/core/models";

@Component({
  selector: "app-search",
  standalone: true,
  templateUrl: "./search.component.html",
  styleUrl: "./search.component.scss",
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, FormsModule],
})
export class SearchComponent {
  private readonly loaderVariants = [
    "search-loader--10",
    "search-loader--11",
    "search-loader--12",
    "search-loader--14",
    "search-loader--18",
    "search-loader--19",
  ] as const;
  private readonly searchInput = viewChild<ElementRef<HTMLInputElement>>(
    "searchInput"
  );
  protected readonly query = signal("");
  protected readonly results = signal<Book[]>([]);
  protected readonly total = signal(0);
  protected readonly searchOpen = signal(false);
  protected readonly isSearching = signal(false);
  protected readonly loaderVariant = signal<string>(this.randomLoaderVariant());

  constructor(
    protected readonly catalog: BookCatalogService,
    private readonly library: LibraryService
  ) {}

  openSearch() {
    this.searchOpen.set(true);
    requestAnimationFrame(() => {
      this.searchInput()?.nativeElement.focus();
      this.searchInput()?.nativeElement.select();
    });
  }

  closeSearch() {
    this.searchOpen.set(false);
  }

  @HostListener("window:keydown.escape")
  handleEscapeKey() {
    if (!this.searchOpen()) {
      return;
    }

    this.closeSearch();
  }

  async runSearch() {
    const value = this.query().trim();
    if (!value) {
      this.results.set([]);
      this.total.set(0);
      this.searchOpen.set(false);
      this.isSearching.set(false);
      return;
    }

    this.loaderVariant.set(this.randomLoaderVariant());
    this.searchOpen.set(false);
    this.isSearching.set(true);
    try {
      const response = await this.catalog.search(value, 12);
      this.results.set(response.items);
      this.total.set(response.total);
    } finally {
      this.isSearching.set(false);
    }
  }

  addToLibrary(book: Book) {
    this.library.addBook(book);
  }

  private randomLoaderVariant() {
    const index = Math.floor(Math.random() * this.loaderVariants.length);
    return this.loaderVariants[index];
  }
}
