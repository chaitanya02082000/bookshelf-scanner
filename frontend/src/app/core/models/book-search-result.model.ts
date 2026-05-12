import {Book} from "./book.model";

export interface BookSearchResult {
  query: string;
  total: number;
  items: Book[];
}
