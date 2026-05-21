import { ApplicationConfig, provideExperimentalZonelessChangeDetection } from "@angular/core";
import { provideHttpClient } from "@angular/common/http";
import { provideRouter } from "@angular/router";
import { routes } from "./app.routes";
import { Capacitor } from "@capacitor/core";

import { provideAuth0 } from '@auth0/auth0-angular';

const isNative = Capacitor.isNativePlatform();
const nativeScheme = "com.gmail.chaitanyagithub0208";
const auth0Domain = "dev-kdeoxnytvveh762k.us.auth0.com";
const redirectUri = isNative
  ? `${nativeScheme}://${auth0Domain}/capacitor/${nativeScheme}/callback`
  : window.location.origin;
export const appConfig: ApplicationConfig = {
    providers: [provideRouter(routes), provideHttpClient(), provideExperimentalZonelessChangeDetection(),
    provideAuth0({
        domain: "dev-kdeoxnytvveh762k.us.auth0.com",
        clientId: "crK6gn79cUaYckp3DArilapQ5oCP2wYZ",
        cacheLocation: "localstorage",
        useRefreshTokens: true,
        useRefreshTokensFallback: false,
        authorizationParams: {
            redirect_uri: redirectUri,
            audience: "bookshelf",
            scope: "openid profile email offline_access",
        },
    }),

    ],
};
