import { ChangeDetectionStrategy, Component } from "@angular/core";
import { RouterOutlet } from "@angular/router";


import { inject } from '@angular/core';
import { AuthService } from '@auth0/auth0-angular';
import { CommonModule } from '@angular/common';
@Component({
    selector: "app-root",
    standalone: true,
    templateUrl: './app.component.html',
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [RouterOutlet, CommonModule,],
})
export class AppComponent {

    protected readonly window = window;
    protected auth = inject(AuthService);
}
