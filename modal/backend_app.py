from __future__ import annotations

import os
import sys

import modal

from common import common_volumes, image


app = modal.App("bookshelf-scanner-backend")


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    scaledown_window=15 * 60,
    volumes=common_volumes,
    secrets=[modal.Secret.from_name("bookshelf-backend-secrets")],
)
@modal.asgi_app()
def fastapi_app():
    backend_root = "/root/bookshelf-scanner/backend"
    ai_root = "/root/bookshelf-scanner/ai/src"

    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    if ai_root not in sys.path:
        sys.path.insert(0, ai_root)

    os.makedirs(f"{backend_root}/output/segmentation", exist_ok=True)

    from src.main import app as web_app

    return web_app
