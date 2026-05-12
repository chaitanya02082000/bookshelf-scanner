import {ChangeDetectionStrategy, Component, computed} from "@angular/core";
import {CommonModule} from "@angular/common";
import {inject} from "@angular/core";
import {LibraryService} from "@/core/services";

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
  protected readonly books = this.libraryService.books;
  protected readonly hasBooks = computed(() => this.books().length > 0);

  removeBook(id: string) {
    this.libraryService.removeBook(id);
  }
}
