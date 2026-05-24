from __future__ import annotations

import modal


notebook_volume = modal.Volume.from_name(
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
        "amzpy==1.0.0",
        "ultralytics==8.3.47",
        "opencv-contrib-python==4.10.0.84",
        "huggingface-hub==0.26.5",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
    )
    .add_local_dir(
        "backend/src",
        remote_path="/root/bookshelf-scanner/backend/src",
        copy=True,
    )
    .add_local_file(
        "backend/pyproject.toml",
        remote_path="/root/bookshelf-scanner/backend/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        "backend/.env.example",
        remote_path="/root/bookshelf-scanner/backend/.env.example",
        copy=True,
    )
    .add_local_dir(
        "ai/src",
        remote_path="/root/bookshelf-scanner/ai/src",
        copy=True,
    )
    .add_local_file(
        "ai/pyproject.toml",
        remote_path="/root/bookshelf-scanner/ai/pyproject.toml",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/root/bookshelf-scanner/backend:/root/bookshelf-scanner/ai/src",
            "HF_HOME": "/mnt/bookshelf-scanner-model-cache/huggingface",
            "TRANSFORMERS_CACHE": "/mnt/bookshelf-scanner-model-cache/huggingface",
            "TORCH_HOME": "/mnt/bookshelf-scanner-model-cache/torch",
            "ULTRALYTICS_HOME": "/mnt/bookshelf-scanner-model-cache/ultralytics",
        }
    )
    .run_commands(
        "mkdir -p /root/bookshelf-scanner/backend/output/segmentation",
        "mkdir -p /mnt/bookshelf-scanner-model-cache/huggingface",
        "mkdir -p /mnt/bookshelf-scanner-model-cache/torch",
        "mkdir -p /mnt/bookshelf-scanner-model-cache/ultralytics",
    )
)

common_volumes = {
    "/mnt/bookshelf-scanner-data": notebook_volume,
    "/mnt/bookshelf-scanner-model-cache": model_cache_volume,
}
