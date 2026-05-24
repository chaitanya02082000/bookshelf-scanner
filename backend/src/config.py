from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_environment() -> None:
    backend_root = Path(__file__).resolve().parent.parent
    workspace_root = backend_root.parent

    # Load the backend-local .env first, then optionally the workspace-level one.
    load_dotenv(backend_root / ".env", override=False)
    load_dotenv(workspace_root / ".env", override=False)


load_environment()
