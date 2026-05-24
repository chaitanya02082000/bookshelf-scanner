from __future__ import annotations

import modal

from common import common_volumes, image


app = modal.App("bookshelf-scanner-notebooks")


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    volumes=common_volumes,
)
def notebook_image() -> str:
    return "bookshelf-scanner notebook image ready"
