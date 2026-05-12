import {ChangeDetectionStrategy, Component, signal} from "@angular/core";
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
  protected readonly query = signal("");
  protected readonly results = signal<Book[]>([]);
  protected readonly total = signal(0);

  constructor(
    protected readonly catalog: BookCatalogService,
    private readonly library: LibraryService
  ) {}

  async runSearch() {
    const value = this.query().trim();
    const response = await this.catalog.search(value, 12);
    this.results.set(response.items);
    this.total.set(response.total);
  }

  addToLibrary(book: Book) {
    this.library.addBook(book);
  }
}
