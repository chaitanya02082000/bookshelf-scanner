import {ChangeDetectionStrategy, Component} from "@angular/core";
import {CommonModule} from "@angular/common";
import {AuthService} from "@auth0/auth0-angular";
import {Capacitor} from "@capacitor/core";
import {Browser} from "@capacitor/browser";

@Component({
  selector: "app-login",
  standalone: true,
  imports: [CommonModule],
  templateUrl: "./login.component.html",
  styleUrl: "./login.component.scss",
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LoginComponent {
  private readonly auth0Domain = "dev-kdeoxnytvveh762k.us.auth0.com";
  private readonly nativeScheme = "com.gmail.chaitanyagithub0208";
  private readonly redirectUri = Capacitor.isNativePlatform()
    ? `${this.nativeScheme}://${this.auth0Domain}/capacitor/${this.nativeScheme}/callback`
    : window.location.origin;

  constructor(public readonly auth: AuthService) {}

  login() {
    this.auth.loginWithRedirect({
      authorizationParams: {
        redirect_uri: this.redirectUri,
      },
      async openUrl(url: string) {
        if (Capacitor.isNativePlatform()) {
          await Browser.open({url, windowName: "_self"});
          return;
        }
        window.location.assign(url);
      },
    });
  }

  signup() {
    this.auth.loginWithRedirect({
      authorizationParams: {
        redirect_uri: this.redirectUri,
        screen_hint: "signup",
      },
      async openUrl(url: string) {
        if (Capacitor.isNativePlatform()) {
          await Browser.open({url, windowName: "_self"});
          return;
        }
        window.location.assign(url);
      },
    });
  }
}
