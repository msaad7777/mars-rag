from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import RAG

app = FastAPI(title="Mars mission research assistant", version="0.1.0")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@lru_cache(maxsize=1)
def rag() -> RAG:
    return RAG()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question) -> dict:
    a = rag().ask(q.question).to_dict()
    for c in a["contexts"]:
        c["text"] = c["text"][:400]
    return a
