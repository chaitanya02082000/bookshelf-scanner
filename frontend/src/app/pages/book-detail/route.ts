import {Routes} from "@angular/router";

export const bookDetailRoutes: Routes = [
  {
    path: "book/:id",
    loadComponent: () =>
      import("./book-detail.component").then((m) => m.BookDetailComponent),
  },
];
