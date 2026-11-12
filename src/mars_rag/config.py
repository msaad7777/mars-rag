"""Central settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MARS_RAG_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
MANIFEST = DATA / "manifest.json"
CHUNKS = DATA / "chunks.jsonl"
CHROMA_DIR = DATA / "chroma"

EMBED_MODEL = os.environ.get("MARS_RAG_EMBED", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("MARS_RAG_RERANK", "Xenova/ms-marco-MiniLM-L-6-v2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("MARS_RAG_LLM", "gpt-oss:20b")

CHUNK_WORDS = int(os.environ.get("MARS_RAG_CHUNK_WORDS", 220))
CHUNK_OVERLAP = int(os.environ.get("MARS_RAG_CHUNK_OVERLAP", 40))
CANDIDATES = 30   # per retriever, before fusion
TOP_K = 5         # passages handed to the LLM
# Cross-encoder logit below which we decline without calling the LLM.
# Tuned on the dev split; see eval/README notes.
DECLINE_THRESHOLD = float(os.environ.get("MARS_RAG_DECLINE_THRESHOLD", -2.0))
