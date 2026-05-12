export interface Book {
  id: string;
  title: string;
  authors: string[];
  coverUrl?: string | null;
  description?: string | null;
  subjects?: string[];
  publishedDate?: string | null;
  pageCount?: number | null;
  isbn?: string | null;
  source?: "openlibrary" | "scan";
}
