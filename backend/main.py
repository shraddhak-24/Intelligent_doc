"""FastAPI app entry. Run with: uvicorn backend.main:app --reload"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.routes import router

app = FastAPI(title="Intelligent Document Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
app.include_router(router)


@app.get("/")
def root():
    return {"service": "Intelligent Document Q&A", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}
