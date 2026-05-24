from __future__ import annotations

import importlib.util
import modal
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
COMMON_PATH = CURRENT_DIR / "common.py"
COMMON_SPEC = importlib.util.spec_from_file_location("bookshelf_scanner_modal_common", COMMON_PATH)
if COMMON_SPEC is None or COMMON_SPEC.loader is None:
    raise RuntimeError(f"Could not load Modal common module from {COMMON_PATH}")
COMMON_MODULE = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(COMMON_MODULE)

common_volumes = COMMON_MODULE.common_volumes
image = COMMON_MODULE.image


app = modal.App("bookshelf-scanner-notebooks")


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    volumes=common_volumes,
)
def notebook_image() -> str:
    return "bookshelf-scanner notebook image ready"
