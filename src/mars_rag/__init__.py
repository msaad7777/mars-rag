"""Mars mission research assistant: RAG over NASA Technical Reports Server documents."""
import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(prog="mars-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="fetch the PDFs listed in data/manifest.json")
    dl.add_argument("--refresh", action="store_true", help="also search NTRS for new reports (changes the corpus)")
    sub.add_parser("ingest", help="chunk, embed and index the PDFs into Chroma")
    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true")
    srv = sub.add_parser("serve", help="run the FastAPI server on 127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    if args.cmd == "download":
        from .ntrs import main as run
        run(refresh=args.refresh)
    elif args.cmd == "ingest":
        from .ingest import main as run
        run()
    elif args.cmd == "ask":
        from .pipeline import RAG
        a = RAG().ask(args.question)
        if args.json:
            print(json.dumps(a.to_dict(), indent=1))
            return
        print(a.answer)
        for c in a.citations:
            print(f"  [{c['n']}] {c['title']}, p.{c['page']}  {c['url']}")
        if a.declined:
            print(f"  (declined: {a.decline_reason})")
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("mars_rag.api:app", host="127.0.0.1", port=args.port)
