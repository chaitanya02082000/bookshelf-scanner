import {Routes} from "@angular/router";

export const libraryRoutes: Routes = [
  {
    path: "library",
    loadComponent: () =>
      import("./library.component").then((m) => m.LibraryComponent),
  },
];
