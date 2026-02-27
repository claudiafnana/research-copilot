<!-- .github/copilot-instructions.md -->
# Copilot / AI agent instructions — Prompt Engineer: tarea 01

Purpose
- Short, targeted guidance so AI coding agents can be productive immediately in this repository.

Repository layout (what matters)
- `prompts:` — primary working area for prompt files and experiment artifacts (currently empty).
- `papers/` — contains PDFs and `paper_catalog.json` used to index papers for prompt/data examples. Example: `papers/paper_catalog.json` is an array of paper entries (currently `[]`).
- `app:` `src:` — placeholder code folders (note: in this workspace these directories are present but empty).
- Top-level files: `requirements.txt` and `README.md` exist but are currently empty.

Key assumptions
- This repo is a prompt engineering experiment space — expect small scripts, prompt files, and paper assets rather than a full application.
- Python virtualenv is used in-repo as `.venv/`. Activate with your shell when running local Python commands:

```bash
source ".venv/bin/activate"
python -m pip install -r requirements.txt   # run only if requirements.txt is populated
```

What to do when asked to implement features
- Prefer adding small, focused Python scripts under `src:` or `app:` and keep a minimal runner (e.g., `src/run.py`).
- If adding dependencies, update `requirements.txt` (one package per line) and include a short usage note in `README.md`.

Project-specific patterns and examples
- Papers indexing: `papers/paper_catalog.json` is the canonical catalog for PDFs in `papers/`. When adding a paper, append an object to the array with keys such as `title`, `filename`, and `tags`.
- Prompts folder: place named prompt files (e.g., `prompts/summarize_paper.txt`) and small metadata (JSON) alongside them if needed.

Developer workflows and commands
- Activate virtualenv: `source ".venv/bin/activate"` (macOS/zsh).
- Install deps (if added): `python -m pip install -r requirements.txt`.
- There are no tests in this repo currently; if you add tests, use `pytest` and document commands in `README.md`.

When editing or generating code
- Keep changes minimal and focused; prefer adding a single new file for each feature.
- Provide a short example usage snippet in the file header or `README.md`.

Integration points and external dependencies
- No external services are configured in the repo. If connecting to external APIs, keep credentials out of the repo and document required env vars in `README.md`.

If you need more context
- Open `papers/` to see existing paper PDFs and `paper_catalog.json`.
- Ask the human owner to describe the desired prompt experiments or how they plan to use the `prompts:` folder.

Fallback behavior for the agent
- If files or instructions are missing (empty README/requirements), make minimal, reversible changes (create new `src/` runner or example prompt) and leave clear commit messages.

Feedback request
- If anything here is unclear or you want agent behavior to be more opinionated (formatting style, testing policy, CI hooks), tell me which area to expand.
