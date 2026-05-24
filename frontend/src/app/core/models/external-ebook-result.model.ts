export interface ExternalEbookResult {
  id: string;
  title: string;
  author?: string | null;
  publisher?: string | null;
  year?: string | null;
  language?: string | null;
  pages?: string | null;
  size?: string | null;
  extension?: string | null;
  source?: string | null;
  score?: number | null;
  coverUrl?: string | null;
  downloadUrl?: string | null;
  resolvedDownloadLink?: string | null;
}

export interface ExternalEbookSearchResult {
  query: string;
  total: number;
  items: ExternalEbookResult[];
}
