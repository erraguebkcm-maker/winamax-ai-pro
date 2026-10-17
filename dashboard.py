"""Génère le tableau de bord mobile (une seule page HTML, sans serveur) à partir du rapport."""
import json
from pathlib import Path

TEMPLATE = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")


def build(report: dict, out_dir: Path):
    payload = json.dumps(report, ensure_ascii=False, default=str).replace("</", "<\\/")
    inner = TEMPLATE.replace("__DATA__", payload)
    # version « page web » complète (GitHub Pages, ouverture directe sur le téléphone)
    page = ("<!doctype html><html lang=\"fr\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">"
            "</head><body>" + inner + "</body></html>")
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    (out_dir / "artifact.html").write_text(inner, encoding="utf-8")
