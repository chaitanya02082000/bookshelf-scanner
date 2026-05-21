import {bootstrapApplication} from "@angular/platform-browser";
import {appConfig} from "@/app.config";
import {AppComponent} from "@/app.component";
import {AuthService} from "@auth0/auth0-angular";
import {App} from "@capacitor/app";
import {Browser} from "@capacitor/browser";

const nativeScheme = "com.gmail.chaitanyagithub0208";
const auth0Domain = "dev-kdeoxnytvveh762k.us.auth0.com";
const callbackUri = `${nativeScheme}://${auth0Domain}/capacitor/${nativeScheme}/callback`;

bootstrapApplication(AppComponent, appConfig)
  .then((appRef) => {
    const auth = appRef.injector.get(AuthService);
    App.addListener("appUrlOpen", ({url}: {url: string}) => {
      console.log("Auth0 appUrlOpen", url);
      if (!url || !url.startsWith(callbackUri)) {
        return;
      }
      auth.handleRedirectCallback(url).subscribe({
        next: () => {
          console.log("Auth0 redirect handled");
          Browser.close();
        },
        error: (err) => console.error("Auth0 redirect error", err),
      });
    });
  })
  .catch((err) => console.error(err));
