"""Download a Mars-mission corpus from the NASA Technical Reports Server (ntrs.nasa.gov).

Only records whose NTRS copyright determination is GOV_PUBLIC_USE_PERMITTED are kept:
works of the US Government, cleared by NASA for public use. Everything else is skipped.
The manifest (IDs, titles, licence field, URLs) is committed; the PDFs are not.
"""
from __future__ import annotations

import json
import time

import httpx

from .config import MANIFEST, PDF_DIR

API = "https://ntrs.nasa.gov/api/citations/search"
BASE = "https://ntrs.nasa.gov"
QUERIES = [
    "human mars mission architecture",
    "mars entry descent and landing",
    "mars in-situ resource utilization propellant",
    "mars surface habitat",
    "mars crew radiation exposure",
    "mars ascent vehicle",
    "mars aerocapture",
    "mars sample return",
    "nuclear propulsion mars transit",
    "mars life support closed loop",
    "mars dust environment",
    "mars landing site selection",
]
SKIP_TYPES = {"PRESENTATION", "ABSTRACT", "POSTER"}
PER_QUERY = 7
MAX_BYTES = 20_000_000


def search(client: httpx.Client, q: str) -> list[dict]:
    r = client.get(API, params={"q": q, "page.size": 100})
    r.raise_for_status()
    return r.json()["results"]


def eligible(rec: dict) -> bool:
    return (
        rec.get("copyright", {}).get("determinationType") == "GOV_PUBLIC_USE_PERMITTED"
        and rec.get("distribution") == "PUBLIC"
        and rec.get("downloadsAvailable")
        and rec.get("stiType") not in SKIP_TYPES
        and any(d.get("mimetype") == "application/pdf" for d in rec.get("downloads", []))
    )


def publication_year(rec: dict) -> str:
    """Publication date, not NTRS upload date (old reports were often uploaded decades later)."""
    pubs = [p.get("publicationDate") for p in rec.get("publications", []) if p.get("publicationDate")]
    return (min(pubs) if pubs else rec.get("created") or rec.get("distributionDate") or "")[:4]


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    if MANIFEST.exists():
        manifest = {d["id"]: d for d in json.loads(MANIFEST.read_text())}
    with httpx.Client(timeout=60, follow_redirects=True,
                      headers={"User-Agent": "mars-rag-portfolio/0.1"}) as client:
        for q in QUERIES:
            taken = 0
            for rec in search(client, q):
                if taken >= PER_QUERY:
                    break
                if not eligible(rec):
                    continue
                rid = str(rec["id"])
                if rid in manifest:
                    taken += 1
                    continue
                pdf = next(d for d in rec["downloads"] if d.get("mimetype") == "application/pdf")
                url = BASE + pdf["links"]["pdf"]
                path = PDF_DIR / f"{rid}.pdf"
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                except httpx.HTTPError as e:
                    print(f"  skip {rid}: {e}")
                    continue
                if len(resp.content) > MAX_BYTES or not resp.content.startswith(b"%PDF"):
                    print(f"  skip {rid}: too large or not a PDF")
                    continue
                path.write_bytes(resp.content)
                manifest[rid] = {
                    "id": rid,
                    "title": rec["title"],
                    "year": publication_year(rec),
                    "center": (rec.get("center") or {}).get("code"),
                    "sti_type": rec.get("stiType"),
                    "licence": rec["copyright"]["determinationType"],
                    "url": f"{BASE}/citations/{rid}",
                    "pdf_url": url,
                    "query": q,
                }
                taken += 1
                print(f"[{q}] {rid} {rec['title'][:70]}")
                time.sleep(0.5)  # be polite to NTRS
    MANIFEST.write_text(json.dumps(sorted(manifest.values(), key=lambda d: d["id"]), indent=1))
    print(f"{len(manifest)} documents in manifest")


if __name__ == "__main__":
    main()
