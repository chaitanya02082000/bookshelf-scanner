import {ChangeDetectionStrategy, Component, signal} from "@angular/core";
import {CommonModule} from "@angular/common";
import {ActivatedRoute} from "@angular/router";
import {Book} from "@/core/models";
import {BookCatalogService, LibraryService} from "@/core/services";

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

  private async load() {
    const workKey = this.route.snapshot.paramMap.get("id");
    if (!workKey) {
      this.isLoading.set(false);
      return;
    }
    const summary = await this.catalog.getWorkDetails(`/works/${workKey}`);
    const existing = this.library
      .books()
      .find((item) => item.id === `/works/${workKey}`) ?? {
      id: `/works/${workKey}`,
      title: "Selected Book",
      authors: ["Unknown"],
      coverUrl: null,
      source: "openlibrary",
    };

    this.book.set({...existing, ...summary});
    this.isLoading.set(false);
  }
}
