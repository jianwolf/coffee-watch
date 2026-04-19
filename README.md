# Coffee Watch

I'm a machine learning engineer who loves making and drinking coffee. I pursue top-quality and interesting coffee choices, but I don't have detailed knowledge about choosing roasters, farms, or beans. This project is an agentic LLM system that monitors specialty roasters and helps me make informed buying decisions.

Coffee Watch is a low-frequency monitoring tool that:
- crawls roaster catalogs politely
- tracks first-seen products in SQLite
- evaluates standout coffees with either Gemini or a local MLX-served model
- writes per-roaster reports plus cross-roaster digests

## Highlights
- Polite crawling with `robots.txt` checks, jittered pacing, and a fixed `User-Agent`
- Shared task prompts across Gemini and local MLX runs for cleaner model/runtime comparison
- Structured run outputs: markdown reports plus `.status.json` and `.items.json` sidecars
- Resume mode that retries only missing or failed roaster runs for the current UTC day
- New-products digest built from the last 7 days of discovered coffees
- Config validation, structured logging, and a real pytest suite

## How It Works
1. Load roasters from `config/roasters.json` and denylisted domains from `config/denylist.txt`.
2. Fetch product lists and, when needed, product pages with robots compliance and retry/backoff.
3. Track products in `logs/seen_products.db` and classify “new” coffees by publish date, HTTP metadata, sitemap metadata, or first-seen timestamp.
4. Build a shared prompt from product metadata, page text, and an optional user ask.
5. Generate one report per roaster.
6. Generate digest reports from today’s roaster outputs.

## Code Structure
- `main.py` — thin entrypoint
- `coffee_watch/cli.py` — CLI parsing and config/bootstrap error handling
- `coffee_watch/config.py` — settings model, CLI flags, config precedence, validation
- `coffee_watch/runner.py` — high-level run orchestration (`full`, `resume`, `digest-only`)
- `coffee_watch/roaster_pipeline.py` — per-roaster flow and digest generation
- `coffee_watch/llm_backend.py` — Gemini and MLX backend abstraction
- `coffee_watch/llm.py` — backward-compatible shim around `llm_backend.py`
- `coffee_watch/classify.py` — new-product date resolution and digest filtering
- `coffee_watch/network.py` — HTTP fetching, retries, robots, sitemaps
- `coffee_watch/http_limits.py` — per-host/global concurrency limiter
- `coffee_watch/report_status.py` — structured sidecars for resume and failure tracking
- `coffee_watch/reporting.py` — report, prompt, and sidecar file helpers
- `coffee_watch/seen_products.py` — SQLite first-seen tracker
- `tests/` — pytest coverage for config, classification, report parsing, resume logic, URL utilities, and MLX text sanitization

## Installation

Basic runtime install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Editable install with dev tooling:

```bash
pip install -e .[dev]
```

## Basic Usage

Run with defaults:

```bash
python main.py
```

Show CLI help:

```bash
python main.py --help
```

Run with Gemini explicitly:

```bash
export GEMINI_API_KEY=your_key_here
python main.py --llm-backend gemini --model gemini-3.1-flash-lite-preview --digest-model gemini-3.1-pro-preview
```

Run with a local MLX backend:

```bash
python main.py --llm-backend mlx --mlx-runtime vlm --mlx-model mlx-community/Qwen3.5-122B-A10B-4bit
```

Common modes:

```bash
python main.py --ask "I want decaf"
python main.py --skip-llm
python main.py --fetch-only
python main.py --digest-only
python main.py --resume
python main.py --config config/settings.json --language en
```

## Configuration

Config precedence is:

`CLI flags > JSON config file > built-in defaults`

Example `config/settings.json`:

```json
{
  "language": "zh",
  "user_ask": "I want decaf",
  "llm_backend": "mlx",
  "model": "mlx-community/Qwen3.5-122B-A10B-4bit",
  "digest_model": "mlx-community/Qwen3.5-122B-A10B-4bit",
  "llm_timeout_s": 600.0,
  "llm_temperature": 1.0,
  "llm_max_tokens": 100000,
  "max_llm_attempts": 10,
  "llm_retry_base_delay_s": 1.0,
  "llm_retry_max_delay_s": 30.0,
  "mlx_model": "mlx-community/Qwen3.5-122B-A10B-4bit",
  "mlx_runtime": "vlm",
  "mlx_host": "127.0.0.1",
  "mlx_port": 8080,
  "mlx_startup_timeout_s": 900.0,
  "mlx_trust_remote_code": false,
  "http_timeout_s": 20.0,
  "http_max_retries": 2,
  "jitter_min_s": 0.7,
  "jitter_max_s": 2.0,
  "http_concurrency": 1,
  "per_host_concurrency": 1,
  "sitemap_max_pages": 8,
  "max_products_per_source": 200,
  "page_text_max_chars": 0,
  "batch_page_text_max_chars": 0,
  "log_json_max_chars": 0,
  "fetch_only": false,
  "skip_llm": false,
  "stream_llm_output": true,
  "digest_only": false,
  "resume": false,
  "save_prompt": false,
  "save_pretty_products_json": false,
  "save_raw_products_json": false,
  "save_report": true,
  "new_products_digest": true,
  "seen_db_path": "logs/seen_products.db",
  "roasters_path": "config/roasters.json",
  "denylist_path": "config/denylist.txt",
  "reports_dir": "reports",
  "assets_dir": "logs/assets",
  "log_path": "logs/coffee_watch.log",
  "log_level": "INFO",
  "log_format": "text"
}
```

Backwards-compatible config aliases still work for:
- `gemini_timeout_s` -> `llm_timeout_s`
- `skip_gemini` -> `skip_llm`
- `ask` / `user_asks` / `asks` -> `user_ask`

## Important Behavior Notes
- Default backend is `gemini`.
- `GEMINI_API_KEY` is only needed when `llm_backend=gemini`.
- When `llm_backend=mlx`, `model` and `digest_model` default to `mlx_model` unless explicitly overridden.
- `mlx_runtime=lm` uses `http://<host>:<port>/v1`; `mlx_runtime=vlm` uses `http://<host>:<port>`.
- `stream_llm_output=true` streams local MLX output to the terminal.
- `skip_llm` skips report and digest generation but still performs crawl/classification work.
- `fetch_only` skips page-level LLM evaluation and writes fetch-only reports.
- `digest_only` rebuilds digests from today’s existing reports and does not update `logs/seen_products.db`.
- `resume` retries only missing or failed roaster runs for the current UTC day, then rebuilds digests.
- The new-products digest covers the last 7 days ending on the current UTC run day.
- Scraped product descriptions are wrapped as untrusted text in prompts to reduce prompt-injection risk.

## Outputs

Generated local outputs include:
- `reports/YYYYMMDD-<roaster-slug>.md` — per-roaster markdown report
- `reports/YYYYMMDD-z-digest.md` — full digest
- `reports/YYYYMMDD-z-roaster-digest.md` — roaster scorecard digest
- `reports/YYYYMMDD-z-new-digest.md` — new-products digest, when enabled and non-empty
- `reports/YYYYMMDD-<roaster-slug>.status.json` — structured roaster run status
- `reports/YYYYMMDD-<roaster-slug>.items.json` — structured new-item payload used by digest rebuilds
- `logs/assets/` — prompt captures and raw/pretty product payloads
- `logs/coffee_watch.log` — text or JSON log output
- `logs/seen_products.db` — SQLite seen-product store

## Testing

Run tests:

```bash
pytest -q
```

Quick syntax check:

```bash
python -m py_compile main.py coffee_watch/*.py tests/*.py
```

Tooling config lives in `pyproject.toml` for:
- `pytest`
- `ruff`
- `mypy`

## Prompting Philosophy
- Roaster and digest prompts are intentionally shared across Gemini and local/open-source backends.
- This keeps model/runtime comparison cleaner by reducing prompt drift.
- If backend-specific prompt behavior ever becomes necessary, it should be clearly documented and justified.

## Responsible Use

This is a hobby project intended for low-frequency monitoring and research.

Please use it responsibly:
- Review and follow each site’s terms of service and `robots.txt`.
- Do not bypass paywalls, authentication, access controls, or anti-bot protections.
- If a site owner asks not to be monitored, add the domain to `config/denylist.txt`.

## License

MIT
