export interface LibgenBook {
  id: string;
  title: string;
  author?: string | null;
  publisher?: string | null;
  year?: string | null;
  language?: string | null;
  pages?: string | null;
  size?: string | null;
  extension?: string | null;
  md5?: string | null;
  mirrors: string[];
  torDownloadLink?: string | null;
  resolvedDownloadLink?: string | null;
}

export interface LibgenSearchResult {
  query: string;
  total: number;
  items: LibgenBook[];
}
