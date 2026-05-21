import {Component, OnDestroy, signal} from "@angular/core";
import {CommonModule} from "@angular/common";
import {FormControl, FormGroup, ReactiveFormsModule} from "@angular/forms";
import {Subscription} from "rxjs";
import {BookCatalogService, BookPredictionService, LibraryService} from "@/core/services";
import {Book} from "@/core/models";
import {AuthService} from "@auth0/auth0-angular";

@Component({
  selector: "app-upload",
  standalone: true,
  templateUrl: "./upload.component.html",
  styleUrl: "./upload.component.scss",
  imports: [CommonModule, ReactiveFormsModule],
})
export class UploadComponent implements OnDestroy {
  protected readonly window = window;
  private readonly highConfidenceThreshold = 0.72;
  private predictionSubscription: Subscription | null = null;
  public readonly uploadForm: FormGroup<UploadForm>;
  public readonly selectedFile = signal<File | null>(null);
  public readonly results = signal<string[]>([]);
  public readonly pendingMatches = signal<PendingMatch[]>([]);
  public readonly notifications = signal<NotificationToast[]>([]);
  public readonly pendingSearchCount = signal(0);
  public readonly errorMessage = signal("");
  public readonly isProcessing = signal(false);
  public readonly reviewOpen = signal(false);
  public readonly scanPanelOpen = signal(false);
  public readonly uploadedImageSrc = signal<string | null>(null);
  public readonly predictedImageSrc = signal<string | null>(null);

  constructor(
    private bookPredictionService: BookPredictionService,
    private readonly catalog: BookCatalogService,
    private readonly library: LibraryService,
    public readonly auth: AuthService
  ) {
    this.uploadForm = new FormGroup<UploadForm>({
      image: new FormControl<File | null>(null),
    });
  }

  ngOnDestroy() {
    if (this.isProcessing()) {
      this.cancelProcess();
    }
  }

  handleFileSelected(event: Event) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    this.selectedFile.set((event.target as any).files[0] ?? null);
    this.results.set([]);
    this.pendingMatches.set([]);
    this.pendingSearchCount.set(0);
    this.errorMessage.set("");
    this.isProcessing.set(false);

    // Read the selected file and set the image preview
    const reader = new FileReader();
    reader.onload = () => {
      this.uploadedImageSrc.set(reader.result as string);
    };

    if (this.selectedFile()) {
      reader.readAsDataURL(this.selectedFile()!);
    }
  }

  submitForm() {
    if (!this.selectedFile()) {
      return;
    }

    this.reviewOpen.set(false);
    this.scanPanelOpen.set(false);
    this.predictedImageSrc.set(null);
    this.pendingMatches.set([]);
    this.pendingSearchCount.set(0);
    this.isProcessing.set(true);

    this.predictionSubscription = this.bookPredictionService
      .startPrediction(this.selectedFile()!)
      .subscribe({
        next: (result) => {
          if (result.success && result.data) {
            const imagePayload = this.normalizeImagePayload(result.data);
            if (imagePayload) {
              this.predictedImageSrc.set(imagePayload);
              return;
            }

            this.results.update((results) => [...results, result.data!]);
            this.fetchBookMetadata(result.data!);
          } else {
            this.errorMessage.set(result.error ?? "Unknown error");
            this.isProcessing.set(false);
            this.cancelProcess();
          }
        },
        error: (error) => {
          this.errorMessage.set(error);
          this.isProcessing.set(false);
        },
        complete: () => {
          this.isProcessing.set(false);
        },
      });
  }

  cancelProcess() {
    this.bookPredictionService.cancelPrediction();

    if (this.predictionSubscription) {
      this.predictionSubscription.unsubscribe();
      this.predictionSubscription = null;
    }

    this.isProcessing.set(false);
  }

  private async fetchBookMetadata(line: string) {
    const parsed = this.parseScanLine(line);
    if (!parsed.title) {
      return;
    }

    this.pendingSearchCount.update((count) => count + 1);
    try {
      const query = parsed.author
        ? `${parsed.title} ${parsed.author}`
        : parsed.title;
      const response = await this.catalog.search(query, 6);
      const bestMatch = this.pickBestCandidate(
        parsed,
        response.items,
        this.highConfidenceThreshold
      );

      if (bestMatch && !this.isDuplicate(bestMatch.book)) {
        if (bestMatch.score >= this.highConfidenceThreshold) {
          this.library.addBook(this.decorateFromScan(bestMatch.book, parsed.raw));
          this.pushNotification(
            `Added to library: ${bestMatch.book.title}`
          );
          return;
        }

        this.addPendingMatch({
          parsed,
          candidate: bestMatch.book,
          confidence: bestMatch.score,
          reason: "Low confidence match",
        });
        return;
      }

      if (bestMatch && this.isDuplicate(bestMatch.book)) {
        return;
      }

      this.addPendingMatch({
        parsed,
        candidate: null,
        confidence: 0,
        reason: "No close match found",
      });
    } finally {
      this.pendingSearchCount.update((count) => Math.max(0, count - 1));
    }
  }

  private decorateFromScan(book: Book, raw: string): Book {
    return {
      ...book,
      source: "scan",
      description: book.description ?? `Imported from scan: ${raw}`,
    };
  }

  private parseScanLine(line: string): ParsedScanLine {
    const raw = line.trim();
    const cleaned = raw.replace(/^Book\s+\d+:/i, "").trim();
    if (!cleaned) {
      return {raw, title: ""};
    }

    const match = cleaned.match(/^(.*?)\s+by\s+(.+)$/i);
    if (match) {
      return {
        raw,
        title: match[1].trim(),
        author: match[2].trim(),
      };
    }

    return {raw, title: cleaned};
  }

  private pickBestCandidate(
    parsed: ParsedScanLine,
    candidates: Book[],
    minScore: number
  ) {
    if (!candidates.length) {
      return null;
    }

    let best: {book: Book; score: number} | null = null;
    for (const book of candidates) {
      const score = this.scoreCandidate(parsed, book);
      if (!best || score > best.score) {
        best = {book, score};
      }
    }

    if (!best || best.score < minScore * 0.6) {
      return null;
    }

    return best;
  }

  private scoreCandidate(parsed: ParsedScanLine, book: Book) {
    const titleScore = this.similarity(
      this.normalizeTitle(parsed.title),
      this.normalizeTitle(book.title)
    );
    if (!parsed.author) {
      return titleScore;
    }

    const authorScore = this.bestAuthorScore(parsed.author, book.authors ?? []);
    const weighted = titleScore * 0.7 + authorScore * 0.3;
    if (authorScore < 0.25) {
      return weighted * 0.85;
    }

    return weighted;
  }

  private bestAuthorScore(author: string, candidates: string[]) {
    if (!candidates.length) {
      return 0;
    }

    const normalizedAuthor = this.normalizeAuthor(author);
    return Math.max(
      ...candidates.map((candidate) =>
        this.similarity(normalizedAuthor, this.normalizeAuthor(candidate))
      )
    );
  }

  private similarity(a: string, b: string) {
    if (!a || !b) {
      return 0;
    }
    if (a === b) {
      return 1;
    }

    const aTokens = new Set(a.split(" ").filter(Boolean));
    const bTokens = new Set(b.split(" ").filter(Boolean));
    const intersection = [...aTokens].filter((token) => bTokens.has(token)).length;
    const max = Math.max(aTokens.size, bTokens.size);
    const baseScore = max ? intersection / max : 0;

    if (a.includes(b) || b.includes(a)) {
      return Math.min(1, baseScore + 0.15);
    }

    return baseScore;
  }

  private normalizeTitle(value: string) {
    return this.normalizeText(value).replace(
      /(\bvolume\b|\bvol\b|\bbook\b|\bpart\b)?\s*#?\d+$/i,
      ""
    );
  }

  private normalizeAuthor(value: string) {
    return this.normalizeText(value);
  }

  private normalizeText(value: string) {
    return value
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  private isDuplicate(candidate: Book) {
    const libraryBooks = this.library.books();
    if (libraryBooks.find((book) => book.id === candidate.id)) {
      return true;
    }

    const candidateTitle = this.normalizeTitle(candidate.title);
    const candidateAuthor = this.normalizeAuthor(
      candidate.authors?.[0] ?? ""
    );
    return libraryBooks.some((book) => {
      const titleMatch =
        this.normalizeTitle(book.title) === candidateTitle && candidateTitle;
      if (!titleMatch) {
        return false;
      }
      if (!candidateAuthor) {
        return true;
      }
      return (
        this.normalizeAuthor(book.authors?.[0] ?? "") === candidateAuthor
      );
    });
  }

  private addPendingMatch(entry: PendingMatchInput) {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    this.pendingMatches.update((matches) => [
      {
        id,
        raw: entry.parsed.raw,
        parsedTitle: entry.parsed.title,
        parsedAuthor: entry.parsed.author ?? null,
        candidate: entry.candidate,
        confidence: entry.confidence,
        reason: entry.reason,
      },
      ...matches,
    ]);
    this.reviewOpen.set(true);
  }

  addPendingToLibrary(entry: PendingMatch) {
    if (entry.candidate && !this.isDuplicate(entry.candidate)) {
      this.library.addBook(this.decorateFromScan(entry.candidate, entry.raw));
      this.pushNotification(`Added to library: ${entry.candidate.title}`);
    }
    this.removePending(entry.id);
  }

  removePending(id: string) {
    this.pendingMatches.update((matches) => {
      const next = matches.filter((match) => match.id !== id);
      if (!next.length) {
        this.reviewOpen.set(false);
      }
      return next;
    });
  }

  openReview() {
    if (!this.pendingMatches().length) {
      return;
    }
    this.reviewOpen.set(true);
  }

  closeReview() {
    this.reviewOpen.set(false);
  }

  openScanPanel() {
    this.scanPanelOpen.set(true);
  }

  closeScanPanel() {
    this.scanPanelOpen.set(false);
  }

  private pushNotification(message: string) {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    this.notifications.update((current) => [
      {id, message},
      ...current,
    ]);
    setTimeout(() => this.dismissNotification(id), 4200);
  }

  dismissNotification(id: string) {
    this.notifications.update((current) =>
      current.filter((notification) => notification.id !== id)
    );
  }

  private normalizeImagePayload(data: string) {
    if (data.startsWith("data:image/")) {
      return data;
    }

    if (!this.looksLikeBase64(data)) {
      return null;
    }

    const mime = this.guessBase64MimeType(data);
    return `data:${mime};base64,${data}`;
  }

  private looksLikeBase64(data: string) {
    if (data.length < 200) {
      return false;
    }

    if (data.includes(" ") || data.includes("\n") || data.includes("\r")) {
      return false;
    }

    return /^[A-Za-z0-9+/=]+$/.test(data);
  }

  private guessBase64MimeType(data: string) {
    if (data.startsWith("iVBOR")) {
      return "image/png";
    }

    if (data.startsWith("/9j/")) {
      return "image/jpeg";
    }

    return "image/jpeg";
  }
}

interface UploadForm {
  image: FormControl<File | null>;
}

interface ParsedScanLine {
  raw: string;
  title: string;
  author?: string;
}

interface PendingMatchInput {
  parsed: ParsedScanLine;
  candidate: Book | null;
  confidence: number;
  reason: string;
}

interface PendingMatch {
  id: string;
  raw: string;
  parsedTitle: string;
  parsedAuthor: string | null;
  candidate: Book | null;
  confidence: number;
  reason: string;
}

interface NotificationToast {
  id: string;
  message: string;
}
