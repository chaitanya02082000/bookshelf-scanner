import {Routes} from "@angular/router";
import {authGuard} from "@/core/guards";
import {loginRoutes, uploadRoutes} from "@/pages";

export const routes: Routes = [
  ...loginRoutes,
  {
    path: "",
    canActivate: [authGuard],
    children: [...uploadRoutes],
  },
  {
    path: "**",
    redirectTo: "",
  },
];
