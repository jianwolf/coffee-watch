# Repository Guidelines

## Project Purpose
- This repo supports real coffee tracking and serves as a public showcase of MLE + backend engineering skills.
- The codebase is intentionally structured to compare hosted Gemini against local MLX-served models while keeping the task prompt shared.

## Project Structure & Module Organization
- Source code lives in `coffee_watch/`.
- Entry point is `main.py`, which calls `coffee_watch/cli.py`.
- Top-level orchestration lives in `coffee_watch/runner.py`.
- Per-roaster scraping, classification, report writing, and digest generation live in `coffee_watch/roaster_pipeline.py`.
- LLM transport selection lives in `coffee_watch/llm_backend.py`; `coffee_watch/llm.py` is a backward-compatible shim.
- New-product date resolution and digest-only filtering live in `coffee_watch/classify.py`.
- HTTP, retries, robots.txt handling, and sitemap fetching live in `coffee_watch/network.py`.
- Per-host concurrency control lives in `coffee_watch/http_limits.py`.
- Structured resume/digest state lives in `coffee_watch/report_status.py` and `coffee_watch/reporting.py`.
- Seen-products tracking lives in `coffee_watch/seen_products.py` (SQLite).
- Config files are under `config/` (for example `config/roasters.json` and `config/denylist.txt`).
- Generated outputs go to `reports/` and `logs/`.
- Tests live under `tests/`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create and activate a virtualenv.
- `pip install -r requirements.txt` — install runtime dependencies.
- `pip install -e .[dev]` — install the project with test and lint tooling from `pyproject.toml`.
- `python main.py` — run with default settings.
- `python main.py --help` — view CLI flags.
- `python main.py --digest-only` — regenerate digests from today’s existing roaster reports only.
- `python main.py --resume` — retry only missing/failed roaster reports for today, then rebuild digests.
- `python main.py --llm-backend mlx --mlx-runtime vlm --mlx-model <model>` — run against a local MLX server target.
- `pytest -q` — run the test suite.
- `python -m py_compile main.py coffee_watch/*.py tests/*.py` — quick syntax sanity check.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and type hints where helpful.
- Keep functions small and single-purpose; prefer descriptive names.
- Preserve the current module split instead of moving logic back into `runner.py`.
- File naming for outputs uses UTC date prefixes such as `YYYYMMDD-roaster-slug.md`, `YYYYMMDD-z-digest.md`, `YYYYMMDD-z-roaster-digest.md`, and `YYYYMMDD-z-new-digest.md`.
- Roaster status and item sidecars use `YYYYMMDD-roaster-slug.status.json` and `YYYYMMDD-roaster-slug.items.json`.
- Avoid introducing non-ASCII text unless the file already uses it.

## Prompt Comparison Policy
- Keep the main task prompt shared across Gemini and local/open-source models unless the user explicitly asks to diverge.
- Backend-specific differences should stay in runtime and transport code, not in the task wording.
- Scraped product descriptions are treated as untrusted text in prompts; preserve that safety boundary when editing prompt builders.

## Testing Guidelines
- Use `pytest -q` as the default verification path.
- Add tests under `tests/` for behavior changes in config parsing, report parsing, resume logic, classification, or LLM text sanitization.
- Keep `python -m py_compile ...` as a fast secondary sanity check for simple edits.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and capitalized, for example `Refactor runner and harden report flow`.
- PRs should include a brief summary, rationale, and runtime notes when behavior or flags change.
- Update `README.md` when user-facing behavior, CLI flags, outputs, or defaults change.

## Security & Configuration Tips
- Secrets come from environment variables such as `GEMINI_API_KEY`; never commit secrets.
- The crawler respects `robots.txt` and uses the fixed project `User-Agent`; do not weaken that behavior.
- Use `config/denylist.txt` to opt out specific domains.
- Seen-products tracking uses `logs/seen_products.db`.
- Per-roaster pacing can be tuned with `jitter_multiplier` in `config/roasters.json`.
- Global and per-host request concurrency are controlled separately via `http_concurrency` and `per_host_concurrency`.
