import {Injectable, inject} from "@angular/core";
import {firstValueFrom} from "rxjs";
import {AuthService} from "@auth0/auth0-angular";
import {Book, Result} from "@/core/models";
import {environment} from "@/../environments/environment";

interface SearchHistoryEventPayload {
  query: string;
  source?: string;
  selectedBookId?: string;
  selectedTitle?: string;
  selectedAuthors?: string[];
  selectedSubjects?: string[];
}

@Injectable({
  providedIn: "root",
})
export class SearchHistoryService {
  private readonly auth = inject(AuthService);
  private readonly apiUrl = environment.apiUrl;

  async recordSearch(query: string, source: string, selectedBook?: Book) {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      return;
    }

    const token = await firstValueFrom(this.auth.getAccessTokenSilently());
    const payload: SearchHistoryEventPayload = {
      query: trimmedQuery,
      source,
      selectedBookId: selectedBook?.id,
      selectedTitle: selectedBook?.title,
      selectedAuthors: selectedBook?.authors,
      selectedSubjects: selectedBook?.subjects,
    };

    await fetch(`${this.apiUrl}/search/history`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "1",
      },
      body: JSON.stringify(payload),
    }).catch(() => null as Result | null);
  }
}
