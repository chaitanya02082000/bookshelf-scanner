from __future__ import annotations

import os

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


bearer_scheme = HTTPBearer(auto_error=True)


class AuthenticatedUser:
    def __init__(self, auth0_user_id: str, email: str | None = None) -> None:
        self.auth0_user_id = auth0_user_id
        self.email = email


class AuthService:
    def __init__(self) -> None:
        self.domain = os.getenv(
            "AUTH0_DOMAIN", "dev-kdeoxnytvveh762k.us.auth0.com"
        )
        self.audience = os.getenv("AUTH0_AUDIENCE", "bookshelf")
        self.issuer = f"https://{self.domain}/"
        self.jwks_client = jwt.PyJWKClient(
            f"https://{self.domain}/.well-known/jwks.json"
        )

    def verify_token(self, token: str) -> AuthenticatedUser:
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.issuer,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            ) from exc

        sub = payload.get("sub")
        if not sub:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing user identity",
            )

        return AuthenticatedUser(auth0_user_id=sub, email=payload.get("email"))


auth_service = AuthService()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> AuthenticatedUser:
    return auth_service.verify_token(credentials.credentials)
