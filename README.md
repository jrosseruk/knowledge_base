# knowledge_base

A personal, notebook-driven ReadTheDocs-style site for notes on **maths** and
**AI**. Built with [Jupyter Book](https://jupyterbook.org) + the
[Sphinx Book theme](https://sphinx-book-theme.readthedocs.io), deployed to
GitHub Pages.

Live site: <https://jrosseruk.github.io/knowledge_base/>

## Repo layout

```
.
├── _config.yml              # Jupyter Book config
├── _toc.yml                 # table of contents (which pages appear, in what order)
├── intro.md                 # landing page (timeline auto-generated from kb_* front-matter)
├── _ext/timeline.py         # the timeline Sphinx extension
├── content/
│   ├── sae/                 # SAE speedrun: 00 primer → 01 superposition → 02 JumpReLU
│   │                        #   → 03 Neuronpedia dashboard → 04 train on Gemma 3 1B
│   ├── training-data-attribution/
│   └── ml/
├── references.md            # bibliography page (references.bib)
├── requirements.txt
└── .github/workflows/deploy.yml
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full rebuild (executes notebooks, writes HTML to _build/html/)
jupyter-book build .

# Faster iteration (skip execution, use cached outputs)
jupyter-book build . --builder html

# Open the result
open _build/html/index.html
```

If a notebook errors during execution the build will fail — fix the notebook
(or set `execute.allow_errors: true` in `_config.yml` if you *want* errors to
appear in the rendered output).

## Adding a new page

1. Drop a notebook or a MyST-Markdown file under `content/<section>/`.
2. Add its path (without extension) to `_toc.yml`.
3. `jupyter-book build .` to preview locally.
4. Commit and push — GitHub Actions rebuilds and deploys.

To add a new top-level section, add another `- caption: ...` entry with its own
`chapters:` list in `_toc.yml`.

## Writing notebooks

- Write maths with standard LaTeX in Markdown cells: inline `$...$`, display
  `$$...$$`. MyST extensions (`amsmath`, `dollarmath`) are enabled in
  `_config.yml`.
- For admonitions, grids, cross-references, etc. see the
  [MyST cheat sheet](https://jupyterbook.org/reference/cheatsheet.html).
- Keep notebooks deterministic — seed your RNGs. The build caches executed
  notebooks, so re-builds are fast as long as the source doesn't change.

## Deployment

Pushes to `main` trigger `.github/workflows/deploy.yml`, which:

1. Installs deps.
2. Runs `jupyter-book build .` (with a cache for executed notebooks).
3. Publishes `_build/html/` to GitHub Pages.

**One-time setup on GitHub:** in the repo's *Settings → Pages*, set the source
to **GitHub Actions**.

## Features

### Live, runnable code cells (Thebe + Binder)

Every notebook page has a `▶ live code` item in the rocket-icon menu at the
top-right. Clicking it boots a Binder kernel in the background (~30s cold
start) and each code cell gains *run* / *restart* controls, so readers can
mutate variables and re-run in place.

- Enabled by `launch_buttons.thebe: true` in `_config.yml`.
- Binder clones this repo and installs `requirements.txt` into the kernel —
  **so Thebe will not work while the repo is private**. Same root cause as the
  Colab rocket.
- Future upgrade: swap Binder for JupyterLite / Pyodide so the kernel runs
  in the reader's browser (no server, faster, but more build plumbing).

### Citations & bibliography (`sphinxcontrib-bibtex`)

A single `references.bib` at the repo root is the canonical bibliography.
Inside any Markdown cell or `.md` file:

```markdown
{cite:t}`koh2017understanding` show that…   ← "Koh & Liang (2017) show that…"
…as shown empirically {cite:p}`grosse2023studying`.   ← "…(Grosse et al., 2023)"
```

A per-page references section at the bottom of a page:

````markdown
## References

```{bibliography}
:filter: docname in docnames
```
````

Inline citations get a native browser tooltip on hover (from the `title=`
attribute) and link back to the global references page.

### Comments (giscus / GitHub Discussions)

Every article page has a hidden comments block at the bottom. To switch it on:

1. Make the repo public (giscus doesn't support private repos).
2. Repo *Settings → General →* tick **Discussions**.
3. Install the giscus app: <https://github.com/apps/giscus>.
4. Go to <https://giscus.app>, enter the repo, copy the generated
   `data-repo-id` and `data-category-id`.
5. In `_config.yml`, under `sphinx.config.html_context`:
   - flip `comments_enabled` to `true`,
   - paste in `giscus_repo_id` and `giscus_category_id`.
6. Rebuild and push.

That's it — readers can now comment on any page using their GitHub account,
and the thread is stored as a GitHub Discussion on the repo.
