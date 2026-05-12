import {Injectable, signal} from "@angular/core";
import {Book} from "@/core/models";

const STORAGE_KEY = "bookscanner.library.v1";

@Injectable({
  providedIn: "root",
})
export class LibraryService {
  private readonly libraryState = signal<Book[]>([]);
  readonly books = this.libraryState.asReadonly();

  constructor() {
    this.load();
  }

  addBook(book: Book) {
    const existing = this.libraryState().find((item) => item.id === book.id);
    if (existing) {
      return;
    }
    this.libraryState.update((state) => [
      {...book, source: book.source ?? "scan"},
      ...state,
    ]);
    this.persist();
  }

  removeBook(id: string) {
    this.libraryState.update((state) =>
      state.filter((book) => book.id !== id)
    );
    this.persist();
  }

  clear() {
    this.libraryState.set([]);
    this.persist();
  }

  hydrate(books: Book[]) {
    this.libraryState.set(books);
    this.persist();
  }

  private load() {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return;
    }
    try {
      const parsed = JSON.parse(raw) as Book[];
      if (parsed) {
        this.libraryState.set(parsed);
      }
    } catch {
      this.libraryState.set([]);
    }
  }

  private persist() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.libraryState()));
  }
}
