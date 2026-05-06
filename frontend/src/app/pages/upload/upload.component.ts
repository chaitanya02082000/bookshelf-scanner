import {Component, OnDestroy, signal} from "@angular/core";
import {CommonModule} from "@angular/common";
import {FormControl, FormGroup, ReactiveFormsModule} from "@angular/forms";
import {Subscription} from "rxjs";
import {BookPredictionService} from "@/core/services";
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
  private predictionSubscription: Subscription | null = null;
  public readonly uploadForm: FormGroup<UploadForm>;
  public readonly selectedFile = signal<File | null>(null);
  public readonly results = signal<string[]>([]);
  public readonly errorMessage = signal("");
  public readonly isProcessing = signal(false);
  public readonly uploadedImageSrc = signal<string | null>(null);
  public readonly predictedImageSrc = signal<string | null>(null);

  constructor(
    private bookPredictionService: BookPredictionService,
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

    this.isProcessing.set(true);

    this.predictionSubscription = this.bookPredictionService
      .startPrediction(this.selectedFile()!)
      .subscribe({
        next: (result) => {
          if (result.success && result.data) {
            // Determine the type based on the data (image or result)
            if (this.predictedImageSrc() == null) {
              // First success response is the image
              this.predictedImageSrc.set(result.data);
            } else {
              // Subsequent success responses are the results
              this.results.update((results) => [...results, result.data!]);
            }
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
}

interface UploadForm {
  image: FormControl<File | null>;
}
