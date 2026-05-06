import {inject} from "@angular/core";
import {CanActivateFn, Router} from "@angular/router";
import {map, tap} from "rxjs";
import {AuthService} from "@auth0/auth0-angular";

export const authGuard: CanActivateFn = () => {
  const auth = inject(AuthService);
  const router = inject(Router);

  return auth.isAuthenticated$.pipe(
    tap((isAuthenticated) => {
      if (!isAuthenticated) {
        router.navigate(["/login"]);
      }
    }),
    map((isAuthenticated) => isAuthenticated)
  );
};
