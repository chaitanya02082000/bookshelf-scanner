import { ApplicationConfig, provideExperimentalZonelessChangeDetection } from "@angular/core";
import { provideHttpClient } from "@angular/common/http";
import { provideRouter } from "@angular/router";
import { routes } from "./app.routes";

import { provideAuth0 } from '@auth0/auth0-angular';
export const appConfig: ApplicationConfig = {
    providers: [provideRouter(routes), provideHttpClient(), provideExperimentalZonelessChangeDetection(),
    provideAuth0({
        domain: "dev-kdeoxnytvveh762k.us.auth0.com",
        clientId: "crK6gn79cUaYckp3DArilapQ5oCP2wYZ",
        authorizationParams: {
            redirect_uri: window.location.origin,
        },
    }),

    ],
};
