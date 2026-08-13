from html import escape
from pathlib import Path
from typing import Iterable


def write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; color: #222; }}
    h1, h2, h3 {{ margin-top: 1.2em; }}
    table {{ border-collapse: collapse; margin: 16px 0; width: 100%; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f3f3; }}
    .note {{ background: #f7f7f7; border-left: 4px solid #999; padding: 10px 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .plot {{ border: 1px solid #ddd; padding: 12px; background: #fff; }}
    .plot img {{ max-width: 100%; height: auto; display: block; }}
    code {{ background: #f2f2f2; padding: 1px 4px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def definition_list(items: Iterable[tuple[str, str]]) -> str:
    rows = []
    for term, description in items:
        rows.append(f"<dt><strong>{escape(term)}</strong></dt><dd>{escape(description)}</dd>")
    return "<dl>" + "\n".join(rows) + "</dl>"
