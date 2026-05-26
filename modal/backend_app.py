from __future__ import annotations

import os
import sys
from pathlib import Path

import modal


app = modal.App("bookshelf-scanner-backend")

BACKEND_ROOT = "/root/bookshelf-scanner/backend"
AI_ROOT = "/root/bookshelf-scanner/ai"
OUTPUT_DIR = "/vol/output"
MODEL_CACHE_DIR = "/vol/model-cache"

output_volume = modal.Volume.from_name(
    "bookshelf-scanner-notebook-data", create_if_missing=True
)
model_cache_volume = modal.Volume.from_name(
    "bookshelf-scanner-model-cache", create_if_missing=True
)

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        add_python="3.12",
    )
    .apt_install(
        "git",
        "curl",
        "wget",
        "tesseract-ocr",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "fastapi[standard]==0.115.6",
        "transformers==4.47.0",
        "accelerate==0.30.0",
        "sentencepiece==0.2.0",
        "pytesseract==0.3.13",
        "libgen-api-enhanced==1.3",
        "pymongo==4.10.1",
        "PyJWT[crypto]==2.10.1",
        "python-dotenv==1.0.1",
        "sentence-transformers==3.4.1",
        "amzpy==1.0.0",
        "ultralytics==8.3.47",
        "opencv-contrib-python==4.10.0.84",
        "huggingface-hub==0.26.5",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
    )
    .add_local_dir("backend/src", remote_path=f"{BACKEND_ROOT}/src", copy=True)
    .add_local_dir("ai/src", remote_path=f"{AI_ROOT}/src", copy=True)
    .env(
        {
            "PYTHONPATH": f"{BACKEND_ROOT}:{AI_ROOT}/src",
            "BOOKSCANNER_OUTPUT_DIR": OUTPUT_DIR,
            "BOOKSCANNER_PRESERVE_OUTPUTS": "1",
            "HF_HOME": f"{MODEL_CACHE_DIR}/huggingface",
            "TRANSFORMERS_CACHE": f"{MODEL_CACHE_DIR}/huggingface",
            "TORCH_HOME": f"{MODEL_CACHE_DIR}/torch",
            "ULTRALYTICS_HOME": f"{MODEL_CACHE_DIR}/ultralytics",
        }
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    scaledown_window=15 * 60,
    volumes={OUTPUT_DIR: output_volume, MODEL_CACHE_DIR: model_cache_volume},
    secrets=[modal.Secret.from_name("bookshelf-backend-secrets")],
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    if BACKEND_ROOT not in sys.path:
        sys.path.insert(0, BACKEND_ROOT)
    ai_src_root = f"{AI_ROOT}/src"
    if ai_src_root not in sys.path:
        sys.path.insert(0, ai_src_root)

    os.makedirs(f"{OUTPUT_DIR}/segmentation", exist_ok=True)
    os.makedirs(f"{MODEL_CACHE_DIR}/huggingface", exist_ok=True)
    os.makedirs(f"{MODEL_CACHE_DIR}/torch", exist_ok=True)
    os.makedirs(f"{MODEL_CACHE_DIR}/ultralytics", exist_ok=True)

    startup_marker = Path(OUTPUT_DIR) / "_modal_startup.txt"
    startup_marker.write_text(
        "modal backend started\n"
        f"output_dir={OUTPUT_DIR}\n"
        f"model_cache_dir={MODEL_CACHE_DIR}\n",
        encoding="utf-8",
    )
    output_volume.commit()

    from src.main import app as web_app

    web_app.state.output_volume = output_volume
    return web_app
