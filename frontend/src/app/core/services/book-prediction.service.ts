import { Injectable, OnDestroy } from "@angular/core";
import { Observable } from "rxjs";
import { Result } from "@/core/models";

@Injectable({
    providedIn: "root",
})
export class BookPredictionService implements OnDestroy {
    private apiUrl = "https://2e03-34-87-49-83.ngrok-free.app/api";
    private abortController: AbortController | null = null;

    ngOnDestroy() {
        this.cancelPrediction();
    }

    startPrediction(file: File): Observable<Result<string>> {
        const formData = new FormData();
        formData.append("file", file);

        // Create an observable to emit streamed results
        return new Observable<Result<string>>((observer) => {
            this.abortController = new AbortController();

            const fetchData = async () => {
                try {
                    const response = await fetch(`${this.apiUrl}/predict`, {
                        method: "POST",
                        body: formData,
                        signal: this.abortController!.signal,
                    });

                    if (!response.ok) {
                        observer.error(`Server error: ${response.statusText}`);
                        return;
                    }

                    const reader = response.body!.getReader();
                    const decoder = new TextDecoder("utf-8");
                    let buffer = "";

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                            observer.complete();
                            break;
                        }

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split("\n");
                        buffer = lines.pop() || "";

                        for (const line of lines) {
                            if (!line.trim()) {
                                continue;
                            }

                            try {
                                const data: Result<string> = JSON.parse(line);
                                observer.next(data);
                            } catch (e) {
                                observer.error(`Error parsing JSON: ${e}`);
                            }
                        }
                    }
                } catch (error) {
                    if (this.abortController?.signal.aborted) {
                        observer.error("Process canceled by user.");
                    } else {
                        observer.error(`Error: ${error}`);
                    }
                } finally {
                    this.abortController = null;
                }
            };

            fetchData();
        });
    }

    cancelPrediction() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
    }
}
