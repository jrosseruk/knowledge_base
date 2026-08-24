"""Build the HTML token-swap panel for FINDINGS.md.

Shows each probe pair as its actual token sequence, with the tokens shared
between the two prompts greyed and the differing run highlighted, so it is
visible that the intervention swaps a described entity rather than a named one.
"""
from __future__ import annotations
import html, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
J, TUNED, MUTED, INK = "#eb6834", "#2a78d6", "#8a8880", "#0b0b0b"


def runs(a, b):
    """Longest common prefix/suffix split, so only the changed run is marked."""
    head = 0
    while head < min(len(a), len(b)) and a[head] == b[head]:
        head += 1
    tail = 0
    while (tail < min(len(a), len(b)) - head
           and a[len(a) - 1 - tail] == b[len(b) - 1 - tail]):
        tail += 1
    return a[:head], a[head:len(a) - tail], b[head:len(b) - tail], a[len(a) - tail:]


def chips(tokens, colour=None):
    out = []
    for token in tokens:
        text = html.escape(token.replace("▁", " ")).replace(" ", "&nbsp;")
        style = (f"background:{colour}22;border:1px solid {colour}66;color:{INK}"
                 if colour else f"background:#00000008;border:1px solid #00000014;color:{MUTED}")
        out.append(f'<span style="display:inline-block;padding:1px 5px;margin:1px;'
                   f'border-radius:4px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
                   f'font-size:12px;{style}">{text}</span>')
    return "".join(out)


def main() -> None:
    items = json.loads((HERE / "prompt_tokens.json").read_text())
    blocks = []
    for item in items:
        head, mid_a, mid_b, tail = runs(item["a"], item["b"])
        blocks.append(f'''
<div style="border:1px solid #00000014;border-radius:8px;padding:10px 12px;margin:8px 0;background:#fcfcfb">
  <div style="font-weight:600;font-size:13px;margin-bottom:6px;color:{INK}">{html.escape(item["name"])}
    <span style="font-weight:400;color:{MUTED}">&nbsp;·&nbsp;unstated bridge
    <code style="color:{J}">{html.escape(item["bridge"])}</code></span></div>
  <div style="line-height:2.0">
    <span style="color:{MUTED};font-size:11px;display:inline-block;width:34px">α=0</span>
    {chips(head)}{chips(mid_a, J)}{chips(tail)}
    <span style="color:{MUTED}">→</span>
    <b style="color:{J}">{html.escape(item["answer"])}</b>
  </div>
  <div style="line-height:2.0">
    <span style="color:{MUTED};font-size:11px;display:inline-block;width:34px">α=1</span>
    {chips(head)}{chips(mid_b, TUNED)}{chips(tail)}
    <span style="color:{MUTED}">→</span>
    <b style="color:{TUNED}">{html.escape(item["counter"])}</b>
  </div>
</div>''')
    legend = (f'<p style="font-size:12px;color:{MUTED};margin:4px 0 10px">'
              f'Grey tokens are shared by both prompts. Only the '
              f'<span style="color:{J}">highlighted</span> run differs, and it never names the '
              f'bridge entity — the model has to retrieve that before it can answer. '
              f'The interpolation runs between the two residual streams these prompts produce.</p>')
    (HERE / "prompt_tokens.html").write_text(legend + "".join(blocks))
    print("wrote prompt_tokens.html")


if __name__ == "__main__":
    main()
