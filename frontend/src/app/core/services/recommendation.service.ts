import {Injectable, inject} from "@angular/core";
import {firstValueFrom} from "rxjs";
import {AuthService} from "@auth0/auth0-angular";
import {BookRecommendation, Result} from "@/core/models";
import {environment} from "@/../environments/environment";

@Injectable({
  providedIn: "root",
})
export class RecommendationService {
  private readonly auth = inject(AuthService);
  private readonly apiUrl = environment.apiUrl;

  async listBooks(limit = 12): Promise<BookRecommendation[]> {
    const token = await firstValueFrom(this.auth.getAccessTokenSilently());
    const url = new URL(`${this.apiUrl}/recommendations/books`);
    url.searchParams.set("limit", String(limit));

    const response = await fetch(url.toString(), {
      headers: {
        Authorization: `Bearer ${token}`,
        "ngrok-skip-browser-warning": "1",
      },
    });

    if (!response.ok) {
      return [];
    }

    const result = (await response.json()) as Result<BookRecommendation[]>;
    return result.data ?? [];
  }
}
