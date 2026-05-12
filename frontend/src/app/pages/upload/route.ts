import {Routes} from "@angular/router";

export const uploadRoutes: Routes = [
  {
    path: "scan",
    loadComponent: () =>
      import("./upload.component").then((m) => m.UploadComponent),
  },
];
