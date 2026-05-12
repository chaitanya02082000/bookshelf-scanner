import {Routes} from "@angular/router";

export const searchRoutes: Routes = [
  {
    path: "search",
    loadComponent: () =>
      import("./search.component").then((m) => m.SearchComponent),
  },
];
