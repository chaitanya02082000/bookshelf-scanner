import {Routes} from "@angular/router";
import {authGuard} from "@/core/guards";
import {
  bookDetailRoutes,
  libraryRoutes,
  loginRoutes,
  searchRoutes,
  uploadRoutes,
} from "@/pages";

export const routes: Routes = [
  ...loginRoutes,
  {
    path: "",
    canActivate: [authGuard],
    children: [
      {path: "", pathMatch: "full", redirectTo: "library"},
      ...uploadRoutes,
      ...libraryRoutes,
      ...searchRoutes,
      ...bookDetailRoutes,
    ],
  },
  {
    path: "**",
    redirectTo: "",
  },
];
