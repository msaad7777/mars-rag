"""Fast unit tests: no models, no network, no index."""
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from mars_rag import config
from mars_rag.components import tokenize
from mars_rag.ingest import clean, usable
from mars_rag.pipeline import PROMPT, RAG, parse_citations


def doc(score: float, n: int = 1) -> Document:
    return Document(page_content=f"Aerocapture saves propellant {n}.", metadata={
        "doc_id": "1", "title": "T", "year": "2000", "page": n, "url": "u", "chunk_id": f"1-p{n}-0",
        "rerank_score": score})


def make_rag(answer: str, docs: list[Document]) -> RAG:
    rag = RAG.__new__(RAG)
    rag.retrieve = lambda q, mode=None, k=config.TOP_K: docs
    rag.chain = PROMPT | FakeListChatModel(responses=[answer])
    return rag


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the MAV's ISRU-propellant?") == ["mav", "s", "isru", "propellant"]


def test_parse_citations_splits_valid_and_invalid():
    valid, invalid = parse_citations("A [1]. B [2, 7]. C [3]", n_sources=5)
    assert valid == {1, 2, 3} and invalid == {7}


def test_clean_rejoins_hyphenation_and_usable_filters_noise():
    assert clean("propel-\nlant  mass") == "propellant mass"
    assert not usable("..... 12 ..... 13 .....")
    assert usable("Aerocapture uses atmospheric drag to slow the vehicle. " * 5)


def test_retrieval_gate_declines_without_calling_llm():
    rag = make_rag("should never be used [1]", [doc(-9.0)])
    a = rag.ask("Who won the World Cup?")
    assert a.declined and a.decline_reason.startswith("retrieval_gate") and "generate" not in a.timings_ms


def test_llm_insufficient_context_declines():
    a = make_rag("INSUFFICIENT_CONTEXT", [doc(3.0)]).ask("q?")
    assert a.declined and a.decline_reason == "llm_insufficient_context"


def test_uncited_answer_is_declined():
    a = make_rag("Aerocapture saves propellant.", [doc(3.0)]).ask("q?")
    assert a.declined and a.decline_reason == "no_valid_citation"


def test_invalid_citation_is_stripped_and_valid_kept():
    a = make_rag("Aerocapture saves propellant [1]. Also X [9].", [doc(3.0)]).ask("q?")
    assert not a.declined
    assert "[9]" not in a.answer and "[1]" in a.answer
    assert [c["n"] for c in a.citations] == [1]
