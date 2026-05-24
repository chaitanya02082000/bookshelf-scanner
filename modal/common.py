from __future__ import annotations

import modal


BACKEND_ROOT = "/root/bookshelf-scanner/backend"
AI_ROOT = "/root/bookshelf-scanner/ai"
OUTPUT_MOUNT_DIR = f"{BACKEND_ROOT}/output"
MODEL_CACHE_MOUNT_DIR = "/vol/bookshelf-scanner-model-cache"


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
    .add_local_dir("backend", remote_path=BACKEND_ROOT, copy=True)
    .add_local_dir("ai", remote_path=AI_ROOT, copy=True)
    .env(
        {
            "PYTHONPATH": f"{BACKEND_ROOT}:{AI_ROOT}/src",
            "BOOKSCANNER_OUTPUT_DIR": OUTPUT_MOUNT_DIR,
            "BOOKSCANNER_PRESERVE_OUTPUTS": "1",
            "HF_HOME": f"{MODEL_CACHE_MOUNT_DIR}/huggingface",
            "TRANSFORMERS_CACHE": f"{MODEL_CACHE_MOUNT_DIR}/huggingface",
            "TORCH_HOME": f"{MODEL_CACHE_MOUNT_DIR}/torch",
            "ULTRALYTICS_HOME": f"{MODEL_CACHE_MOUNT_DIR}/ultralytics",
        }
    )
)

common_volumes = {
    OUTPUT_MOUNT_DIR: notebook_volume,
    MODEL_CACHE_MOUNT_DIR: model_cache_volume,
}
