"""Evaluate the Mars RAG pipeline. Writes eval/results/*.json and prints a summary.

Stages (each can be run alone):
  retrieval  - ablation of dense / BM25 / hybrid / hybrid+re-rank, no LLM
  answers    - run the full pipeline on answerable + unanswerable questions
  ragas      - RAGAS faithfulness, context precision, context recall (local judge)
  citations  - per-sentence citation support check (local judge)
  summary    - combine everything into eval/results/summary.json
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

from mars_rag import config

E = Path(__file__).parent
R = E / "results"
R.mkdir(exist_ok=True)


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def questions():
    return load(E / "questions.jsonl"), load(E / "unanswerable.jsonl")


def stage_retrieval(rag):
    ans, unans = questions()
    out = {}
    for mode in ["dense", "bm25", "hybrid", "bm25_rerank", "hybrid_rerank"]:
        page_hits = chunk_hits = doc_hits = 0
        rr = []
        t = time.perf_counter()
        for q in ans:
            docs = rag.retrieve(q["question"], mode=mode)
            keys = [(d.metadata["doc_id"], d.metadata["page"]) for d in docs]
            gold = (q["gold_doc"], q["gold_page"])
            page_hits += gold in keys
            doc_hits += q["gold_doc"] in [k[0] for k in keys]
            chunk_hits += q["gold_chunk"] in [d.metadata["chunk_id"] for d in docs]
            rr.append(1 / (keys.index(gold) + 1) if gold in keys else 0.0)
        n = len(ans)
        out[mode] = {"n": n, "page_hit@5": round(page_hits / n, 3), "chunk_hit@5": round(chunk_hits / n, 3),
                     "doc_hit@5": round(doc_hits / n, 3), "mrr@5_page": round(sum(rr) / n, 3),
                     "ms_per_query": round((time.perf_counter() - t) * 1000 / n)}
        print(mode, out[mode])
    # Score distributions behind the retrieval decline gate
    top = lambda qs: [rag.retrieve(q["question"])[0].metadata["rerank_score"] for q in qs]
    a_scores, u_scores = top(ans), top(unans)
    sweep = []
    for th in [-6, -4, -3, -2, -1, 0, 1, 2]:
        sweep.append({"threshold": th,
                      "answerable_blocked": round(sum(s < th for s in a_scores) / len(a_scores), 3),
                      "unanswerable_blocked": round(sum(s < th for s in u_scores) / len(u_scores), 3)})
    out["gate_sweep"] = sweep
    out["top_scores"] = {"answerable": a_scores, "unanswerable": u_scores}
    json.dump(out, open(R / "retrieval.json", "w"), indent=1)


def stage_answers(rag):
    ans, unans = questions()
    rows = []
    for q in ans + unans:
        a = rag.ask(q["question"]).to_dict()
        a["answerable"] = "gold_doc" in q
        a["reference"] = q.get("reference")
        a["kind"] = q.get("kind", "answerable")
        rows.append(a)
        print(("DECLINE " if a["declined"] else "ANSWER  ") + q["question"][:80], a["timings_ms"])
    json.dump(rows, open(R / "answers.json", "w"), indent=1)


def judge_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL, reasoning="low",
                      temperature=0, num_ctx=8192)


def stage_ragas():
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

    from mars_rag.components import FastEmbedEmbeddings

    rows = [r for r in json.load(open(R / "answers.json")) if r["answerable"] and not r["declined"]]
    ds = EvaluationDataset.from_list([{
        "user_input": r["question"], "response": r["answer"], "reference": r["reference"],
        "retrieved_contexts": [c["text"] for c in r["contexts"]]} for r in rows])
    llm = LangchainLLMWrapper(judge_llm())
    res = evaluate(ds, metrics=[Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()],
                   llm=llm, embeddings=LangchainEmbeddingsWrapper(FastEmbedEmbeddings(config.EMBED_MODEL)),
                   run_config=RunConfig(timeout=900, max_workers=2, max_retries=2), show_progress=True)
    df = res.to_pandas()
    df.to_json(R / "ragas_rows.json", orient="records", indent=1)
    means = {c: round(float(df[c].mean()), 3) for c in ["faithfulness", "llm_context_precision_with_reference", "context_recall"]}
    means["n"] = len(df)
    means["nan_rows"] = {c: int(df[c].isna().sum()) for c in ["faithfulness", "llm_context_precision_with_reference", "context_recall"]}
    json.dump(means, open(R / "ragas.json", "w"), indent=1)
    print(means)


CITE_PROMPT = """Does the SOURCE text support the CLAIM? Minor paraphrase is fine; the key facts and numbers must be in the source.
Answer with one word: YES or NO.

CLAIM: {claim}

SOURCE:
{source}"""


def stage_citations():
    llm = judge_llm()
    rows = [r for r in json.load(open(R / "answers.json")) if not r["declined"]]
    checks = []
    for r in rows:
        ctx = {c["n"]: c["text"] for c in r["contexts"]}
        for sent in re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", r["answer"]):
            nums = [int(n) for g in re.findall(r"\[([\d,\s]+)\]", sent) for n in re.findall(r"\d+", g)]
            if not nums:
                continue
            claim = re.sub(r"\s*\[[\d,\s]+\]", "", sent).strip()
            for n in nums:
                v = llm.invoke(CITE_PROMPT.format(claim=claim, source=ctx[n])).content.strip().upper()
                checks.append({"question": r["question"], "claim": claim, "cited": n, "supported": v.startswith("YES")})
    sentences = [s for r in rows for s in re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", r["answer"]) if len(s) > 20]
    cited_sentences = [s for s in sentences if re.search(r"\[\d", s)]
    out = {"citations_checked": len(checks),
           "citation_accuracy": round(sum(c["supported"] for c in checks) / max(1, len(checks)), 3),
           "sentence_citation_coverage": round(len(cited_sentences) / max(1, len(sentences)), 3)}
    json.dump({"summary": out, "checks": checks}, open(R / "citations.json", "w"), indent=1)
    print(out)


def stage_summary():
    rows = json.load(open(R / "answers.json"))
    a = [r for r in rows if r["answerable"]]
    u = [r for r in rows if not r["answerable"]]
    lat = sorted(r["timings_ms"].get("total", r["timings_ms"].get("retrieve", 0) + r["timings_ms"].get("generate", 0)) for r in rows)
    reasons = {}
    for r in rows:
        if r["declined"]:
            key = ("answerable:" if r["answerable"] else "unanswerable:") + r["decline_reason"].split(" ")[0]
            reasons[key] = reasons.get(key, 0) + 1
    summary = {
        "corpus": {"documents": len(json.load(open(config.MANIFEST))), "chunks": sum(1 for _ in open(config.CHUNKS))},
        "questions": {"answerable": len(a), "unanswerable": len(u)},
        "decline": {
            "unanswerable_correctly_declined": round(sum(r["declined"] for r in u) / len(u), 3),
            "answerable_wrongly_declined": round(sum(r["declined"] for r in a) / len(a), 3),
            "by_reason": reasons},
        "latency_ms": {"p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95) - 1], "max": lat[-1]},
        "models": {"llm": config.LLM_MODEL, "embeddings": config.EMBED_MODEL, "reranker": config.RERANK_MODEL},
    }
    for name in ["retrieval", "ragas", "citations"]:
        p = R / f"{name}.json"
        if p.exists():
            d = json.load(open(p))
            summary[name] = d.get("summary", d) if name == "citations" else (
                {k: v for k, v in d.items() if k not in ("top_scores",)} if name == "retrieval" else d)
    json.dump(summary, open(R / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    stages = sys.argv[1:] or ["retrieval", "answers", "ragas", "citations", "summary"]
    rag = None
    if {"retrieval", "answers"} & set(stages):
        from mars_rag.pipeline import RAG
        rag = RAG()
    for s in stages:
        print(f"== {s}")
        fn = globals()[f"stage_{s}"]
        fn(rag) if s in ("retrieval", "answers") else fn()
