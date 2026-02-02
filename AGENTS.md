# Repository Guidelines

## Project Purpose
- This repo supports real coffee tracking and serves as a public showcase of MLE + backend engineering skills.

## Project Structure & Module Organization
- Source code lives in `coffee_watch/` and is organized by responsibility (networking, parsing, prompts, reporting, Gemini helpers).
- Entry point is `main.py`, which calls `coffee_watch/cli.py`.
- Configuration files are under `config/` (e.g., `config/roasters.json`, `config/denylist.txt`).
- Generated artifacts go to `reports/` and `logs/` (both are local outputs).
- Dependencies are listed in `requirements.txt`.
- Seen-products tracking lives in `coffee_watch/seen_products.py` (SQLite) and catalog parsing lives in `coffee_watch/catalog_parsers.py`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create/activate a virtualenv.
- `pip install -r requirements.txt` — install dependencies.
- `python main.py` — run the monitor with defaults.
- `python main.py --help` — view CLI flags.
- `python main.py --digest-only` — generate digest from existing reports only.
- `python main.py --config config/settings.json --language en` — run with config overrides.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and type hints where helpful.
- Keep functions small and single-purpose; prefer descriptive names (`fetch_products_for_roaster`, `build_batch_prompt`).
- File naming: outputs use `YYYYMMDD-roaster-slug.md`, `YYYYMMDD-digest.md`, and `YYYYMMDD-new-digest.md` (UTC date).
- Avoid introducing non-ASCII text unless the file already uses it (e.g., prompt language strings).

## Testing Guidelines
- No formal test suite is currently configured.
- Use `python -m py_compile main.py coffee_watch/*.py` for a quick sanity check.
- If you add tests, place them under `tests/` and document how to run them.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and capitalized (e.g., “Add digest report generation”).
- PRs should include a brief summary, rationale, and any runtime notes (flags used, config changes).
- If behavior changes, call it out explicitly and update `README.md` when user-facing.

## Security & Configuration Tips
- Secrets are read from environment variables (e.g., `GEMINI_API_KEY`); never commit secrets.
- The crawler respects `robots.txt` and uses a fixed User-Agent; keep these defaults intact.
- Use `config/denylist.txt` to opt out specific domains.
- Seen-products tracking uses `logs/seen_products.db` to persist first-seen hashes.
- Per-roaster throttling can be tuned via `jitter_multiplier` in `config/roasters.json`.
