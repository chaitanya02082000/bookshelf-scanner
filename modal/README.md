# Modal Images And Backend

This folder contains:

- a shared custom Modal image with the backend + AI dependencies baked in
- a notebook target using that image
- a proper FastAPI ASGI deployment entrypoint using that same image

## What it does

- Uses a CUDA-enabled PyTorch base image
- Installs backend + AI Python dependencies into the image
- Copies project source into the image with `copy=True`
- Creates a persistent Modal volume mounted at `/mnt/bookshelf-scanner-data`

## Deploy notebook image

From the repository root:

```bash
pip install modal
modal setup
modal deploy modal/notebook_image.py
```

## Deploy backend app

First create a Modal secret named `bookshelf-backend-secrets` with the backend env vars:

- `MONGODB_URI`
- `MONGODB_DB`
- `MONGODB_BOOKS_COLLECTION`
- `AUTH0_DOMAIN`
- `AUTH0_AUDIENCE`

Then deploy:

```bash
modal deploy modal/backend_app.py
```

This deploys the FastAPI backend directly on Modal using the custom image.

## Serve backend in development

For a live development endpoint that reloads when you redeploy the file:

```bash
modal serve modal/backend_app.py
```

This is the fastest way to test the backend on Modal without creating a permanent deployment first.

You can then point your frontend to the served Modal URL plus `/api`.

## Run notebook image in development

To verify the image builds correctly during development:

```bash
modal run modal/notebook_image.py
```

This does not start the backend. It just confirms the image is available and the function can start.

## Notes

- The image copies `backend/src` and `ai/src` into the image, so notebooks and the backend can import project code without relying on a slow mounted source tree.
- The attached volume appears under `/mnt/bookshelf-scanner-data`.
- Model caches persist under `/mnt/bookshelf-scanner-model-cache`.
- Scan outputs and temporary images are written under `/mnt/bookshelf-scanner-data/output`.
- If you change Python dependencies, redeploy the image.
- If you change backend or AI source code, rerun `modal serve` or redeploy the backend app.
