"""Print the Markdown table of NASA reports in data/manifest.json (used in the README)."""
import json
from pathlib import Path

rows = json.loads((Path(__file__).parents[1] / "data" / "manifest.json").read_text())
print("| NTRS ID | Title | Year | Centre | Type |\n|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["query"], r["year"])):
    title = r["title"].replace("|", "/")
    print(f"| [{r['id']}]({r['url']}) | {title} | {r['year']} | {r['center'] or ''} | {r['sti_type'].replace('_', ' ').title()} |")
