import {Routes} from "@angular/router";

export const uploadRoutes: Routes = [
  {
    path: "",
    loadComponent: () =>
      import("./upload.component").then((m) => m.UploadComponent),
  },
];
