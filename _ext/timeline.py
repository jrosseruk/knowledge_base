"""
Sphinx extension: auto-generated timeline directive.

Scans all documents for ``kb_date``, ``kb_tag``, ``kb_title``, and ``kb_desc``
metadata (set via MyST YAML front-matter or notebook-level metadata for .ipynb).
The ``{timeline}`` directive renders a reverse-chronological, year-grouped HTML
timeline using the same markup that ``custom.css`` already styles.
"""

from __future__ import annotations

import html as html_mod
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

_TAG_CLASSES = {
    "ai": "kb-tag-ai",
    "ml": "kb-tag-ml",
    "maths": "kb-tag-maths",
    "paper": "kb-tag-paper",
    "note": "kb-tag-note",
}


def _read_notebook_kb_meta(src_path: Path) -> dict[str, str]:
    """Extract kb_* keys from a .ipynb file's notebook-level metadata."""
    try:
        nb = json.loads(src_path.read_text("utf-8"))
        meta = nb.get("metadata", {})
        return {k: v for k, v in meta.items() if k.startswith("kb_")}
    except Exception:
        return {}


def _collect_entries(app: Sphinx) -> list[dict[str, Any]]:
    """Return all documents that declare timeline metadata, sorted newest-first."""
    env = app.env
    srcdir = Path(env.srcdir)
    entries: list[dict[str, Any]] = []

    for docname in env.found_docs:
        metadata = dict(env.metadata.get(docname, {}))

        # For .ipynb files, MyST-NB doesn't promote custom keys to env.metadata,
        # so we read them directly from the notebook JSON.
        if not any(k.startswith("kb_") for k in metadata):
            for ext in (".ipynb",):
                nb_path = srcdir / (docname + ext)
                if nb_path.exists():
                    metadata.update(_read_notebook_kb_meta(nb_path))
                    break

        raw_date = metadata.get("kb_date")
        if not raw_date:
            continue

        try:
            entry_date = date.fromisoformat(str(raw_date))
        except ValueError:
            logger.warning(
                "timeline: invalid kb_date %r in %s, skipping", raw_date, docname
            )
            continue

        title = metadata.get("kb_title", docname.split("/")[-1].replace("-", " ").title())
        desc = metadata.get("kb_desc", "")
        tag = metadata.get("kb_tag", "note")

        docpath = app.builder.get_relative_uri("intro", docname)

        entries.append(
            {
                "date": entry_date,
                "tag": tag,
                "title": title,
                "desc": desc,
                "href": docpath,
            }
        )

    entries.sort(key=lambda e: e["date"], reverse=True)
    return entries


def _render_timeline(entries: list[dict[str, Any]]) -> str:
    """Produce the full timeline HTML from collected entries."""
    if not entries:
        return "<p><em>No entries yet.</em></p>"

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["date"].year].append(entry)

    parts: list[str] = []
    for year in sorted(grouped, reverse=True):
        parts.append(f"<h2>{year}</h2>")
        parts.append('<ol class="kb-timeline">')

        for i, entry in enumerate(grouped[year]):
            side = "kb-entry--right" if i % 2 == 0 else "kb-entry--left"
            tag_class = _TAG_CLASSES.get(entry["tag"], "kb-tag-note")
            tag_label = html_mod.escape(entry["tag"].upper())
            date_str = entry["date"].strftime("%b %d")
            title = html_mod.escape(entry["title"])
            desc = entry["desc"]

            parts.append(f'  <li class="kb-entry {side}">')
            parts.append(f'    <a class="kb-card" href="{entry["href"]}">')
            parts.append(f'      <span class="kb-date">{date_str}</span>')
            parts.append(f'      <span class="kb-tag {tag_class}">{tag_label}</span>')
            parts.append(f'      <h3 class="kb-title">{title}</h3>')
            parts.append(f'      <p class="kb-desc">{desc}</p>')
            parts.append("    </a>")
            parts.append("  </li>")

        parts.append("</ol>")

    return "\n".join(parts)


class TimelineDirective(Directive):
    """MyST/rST directive that emits the auto-generated timeline."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self) -> list[nodes.Node]:
        env = self.state.document.settings.env
        app = env.app
        entries = _collect_entries(app)
        raw_html = _render_timeline(entries)
        node = nodes.raw("", raw_html, format="html")
        return [node]


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_directive("timeline", TimelineDirective)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
