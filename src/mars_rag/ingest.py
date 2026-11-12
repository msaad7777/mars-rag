"""PDF -> pages -> chunks -> embeddings -> Chroma. Also writes chunks.jsonl for BM25 and eval."""
from __future__ import annotations

import json
import re
import shutil
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import config
from .components import FastEmbedEmbeddings

COLLECTION = "ntrs_mars"


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)        # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def usable(text: str) -> bool:
    """Drop OCR noise, tables of dots and near-empty fragments."""
    if len(text) < 200:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / len(text) > 0.6


def load_pages(meta: dict) -> list[Document]:
    path = config.PDF_DIR / f"{meta['id']}.pdf"
    pages = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            try:
                text = clean(page.extract_text() or "")
            except Exception:  # a single broken page should not sink the document
                continue
            if text:
                pages.append(Document(page_content=text, metadata={
                    "doc_id": meta["id"], "title": meta["title"], "year": meta["year"],
                    "url": meta["url"], "page": i + 1}))
    except Exception as e:
        print(f"  unreadable {meta['id']}: {e}")
    return pages


def build_chunks() -> list[Document]:
    manifest = json.loads(config.MANIFEST.read_text())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "])
    chunks: list[Document] = []
    for meta in manifest:
        pages = load_pages(meta)
        kept = 0
        for piece in splitter.split_documents(pages):
            if not usable(piece.page_content):
                continue
            piece.metadata["chunk_id"] = f"{meta['id']}-p{piece.metadata['page']}-{kept}"
            chunks.append(piece)
            kept += 1
        print(f"{meta['id']}: {len(pages)} pages, {kept} chunks")
    return chunks


def main() -> None:
    t0 = time.time()
    chunks = build_chunks()
    with config.CHUNKS.open("w") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.page_content, **c.metadata}) + "\n")
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    store = Chroma(collection_name=COLLECTION, embedding_function=FastEmbedEmbeddings(config.EMBED_MODEL),
                   persist_directory=str(config.CHROMA_DIR), collection_metadata={"hnsw:space": "cosine"})
    batch = 256
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        store.add_documents(part, ids=[c.metadata["chunk_id"] for c in part])
    print(f"indexed {len(chunks)} chunks in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
