import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models import ResultWithData
from .routers import api_router

logging.basicConfig(level=logging.INFO, format="%(levelname)s:    %(message)s")

app = FastAPI()
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root() -> ResultWithData[str]:
    return ResultWithData[str].succeed("API is running")
