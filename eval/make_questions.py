"""Draft answerable eval questions from sampled chunks (one chunk per document).

The LLM drafts a question + reference answer from a single chunk; every draft is then
reviewed by hand and kept, edited or dropped (see eval/questions.jsonl).
"""
import json
import random
import re

from langchain_ollama import ChatOllama

from mars_rag import config
from mars_rag.pipeline import load_chunks

random.seed(7)
PROMPT = """You write evaluation questions for a search system over NASA technical reports.
Read the excerpt and write ONE specific factual question that this excerpt answers, plus a short
reference answer taken from the excerpt. The question must make sense on its own, without seeing
the excerpt (name the system, mission or concept; never say "the excerpt", "this paper" or "the study").
If the excerpt is only references, a table of contents or boilerplate, return {{"skip": true}}.
Return JSON only: {{"question": "...", "answer": "..."}}

Title: {title}
Excerpt:
{text}"""

chunks = load_chunks()
by_doc = {}
for c in chunks:
    if re.search(r"(references|bibliography|acknowledg)", c.page_content[:200], re.I):
        continue
    by_doc.setdefault(c.metadata["doc_id"], []).append(c)
llm = ChatOllama(model=config.LLM_MODEL, reasoning="low", temperature=0, format="json")
out = open("eval/candidates.jsonl", "w")
for doc_id in sorted(by_doc):
    c = random.choice(by_doc[doc_id])
    try:
        r = json.loads(llm.invoke(PROMPT.format(title=c.metadata["title"], text=c.page_content)).content)
    except Exception as e:
        print("fail", doc_id, e); continue
    if r.get("skip") or not r.get("question"):
        continue
    row = {"question": r["question"], "reference": r["answer"], "gold_chunk": c.metadata["chunk_id"],
           "gold_doc": doc_id, "gold_page": c.metadata["page"], "source_text": c.page_content}
    out.write(json.dumps(row) + "\n"); out.flush()
    print(doc_id, "|", r["question"])
