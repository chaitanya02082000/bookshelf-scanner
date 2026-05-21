import {ChangeDetectionStrategy, Component, OnDestroy, signal} from "@angular/core";
import {AsyncPipe, CommonModule} from "@angular/common";
import {NavigationEnd, Router, RouterLink, RouterOutlet} from "@angular/router";
import {Subscription} from "rxjs";
import {AuthService} from "@auth0/auth0-angular";
import {Capacitor} from "@capacitor/core";
@Component({
  selector: "app-root",
  standalone: true,
  templateUrl: "./app.component.html",
  styleUrl: "./app.component.scss",
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, RouterOutlet, RouterLink, AsyncPipe],
})
export class AppComponent implements OnDestroy {
  private readonly routerSubscription: Subscription;
  protected readonly isLoginRoute = signal(false);
  protected readonly window = window;
  private readonly auth0Domain = "dev-kdeoxnytvveh762k.us.auth0.com";
  private readonly nativeScheme = "com.gmail.chaitanyagithub0208";
  protected readonly logoutReturnTo = Capacitor.isNativePlatform()
    ? `${this.nativeScheme}://${this.auth0Domain}/capacitor/${this.nativeScheme}/callback`
    : window.location.origin;

  constructor(
    private readonly router: Router,
    public readonly auth: AuthService
  ) {
    this.isLoginRoute.set(this.router.url.startsWith("/login"));
    this.routerSubscription = this.router.events.subscribe((event) => {
      if (event instanceof NavigationEnd) {
        this.isLoginRoute.set(event.urlAfterRedirects.startsWith("/login"));
      }
    });
  }

  ngOnDestroy() {
    this.routerSubscription.unsubscribe();
  }
}
