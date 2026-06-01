export interface BookComment {
  id: string;
  bookKey: string;
  bookId: string;
  title: string;
  authors: string[];
  isbn?: string | null;
  userDisplayName: string;
  body: string;
  createdAt: string;
  updatedAt: string;
  isOwner: boolean;
}

export interface BookCommentBookPayload {
  bookId: string;
  title: string;
  authors: string[];
  isbn?: string | null;
}

export interface CreateBookCommentRequest extends BookCommentBookPayload {
  body: string;
  userDisplayName?: string | null;
}
