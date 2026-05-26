import {Book} from "./book.model";

export interface BookRecommendation extends Book {
  score: number;
  reason?: string | null;
  matchedAuthors?: string[];
  matchedSubjects?: string[];
  matchedQueries?: string[];
}
