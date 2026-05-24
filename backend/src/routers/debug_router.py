import os
from pathlib import Path

from fastapi import APIRouter

from src.models import ResultWithData


debug_router = APIRouter(prefix="/debug", tags=["debug"])


@debug_router.get("/output")
def inspect_output() -> ResultWithData[dict]:
    output_dir = Path(os.getenv("BOOKSCANNER_OUTPUT_DIR", "output")).resolve()
    exists = output_dir.exists()
    files: list[str] = []
    if exists:
        for path in sorted(output_dir.rglob("*")):
            files.append(str(path.relative_to(output_dir)))

    payload = {
        "outputDir": str(output_dir),
        "exists": exists,
        "files": files,
    }
    return ResultWithData[dict].succeed(payload)
