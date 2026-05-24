import {ChangeDetectionStrategy, Component, signal} from "@angular/core";
import {CommonModule} from "@angular/common";
import {ActivatedRoute} from "@angular/router";
import {Book} from "@/core/models";
import {BookCatalogService, LibraryService} from "@/core/services";
import {formatBookSummaryHtml} from "@/core/utils/book-summary";

@Component({
  selector: "app-book-detail",
  standalone: true,
  templateUrl: "./book-detail.component.html",
  styleUrl: "./book-detail.component.scss",
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule],
})
export class BookDetailComponent {
  protected readonly book = signal<Book | null>(null);
  protected readonly isLoading = signal(true);

  constructor(
    private readonly route: ActivatedRoute,
    private readonly catalog: BookCatalogService,
    private readonly library: LibraryService
  ) {
    this.load();
  }

  addToLibrary() {
    const book = this.book();
    if (book) {
      this.library.addBook(book);
    }
  }

  protected formatSummary(summary?: string | null): string {
    return formatBookSummaryHtml(summary);
  }

  private async load() {
    const workKey = this.route.snapshot.paramMap.get("id");
    if (!workKey) {
      this.isLoading.set(false);
      return;
    }
    const existing = this.library
      .books()
      .find((item) => item.id === `/works/${workKey}`) ?? {
      id: `/works/${workKey}`,
      title: "Selected Book",
      authors: ["Unknown"],
      coverUrl: null,
      source: "openlibrary",
    };
    const fallbackQuery = existing.authors[0]?.trim()
      ? `${existing.title} ${existing.authors[0]}`
      : existing.title;
    const summary = await this.catalog.getWorkDetails(`/works/${workKey}`, fallbackQuery);

    this.book.set({...existing, ...summary});
    this.isLoading.set(false);
  }
}
