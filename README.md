# Coffee Watch

Coffee Watch is a low-frequency specialty coffee catalog scraper. It crawls roaster catalogs politely, tracks first-seen products, and writes normalized JSON that Codex can analyze interactively with the bundled `coffee-scout` skill.

The project is intentionally split into two parts:

```text
scrape roaster catalogs -> normalized JSON/state -> Codex coffee-scout analysis
```

The scraper does not call Gemini, MLX, or any other model API. It preserves evidence and source URLs; the buying judgment happens later in Codex.

## Highlights

- Polite crawling with `robots.txt` checks, jittered pacing, retries, and a fixed `User-Agent`
- Per-host and global HTTP concurrency limits
- Shopify, JSON, and HTML catalog parsing
- SQLite first-seen tracking in `logs/seen_products.db`
- Normalized per-roaster and combined catalog JSON outputs
- Structured status sidecars for resume/retry behavior
- Repo-local `skills/coffee-scout` skill for interactive buying analysis
- Config validation, structured logging, and pytest coverage

## How It Works

1. Load roasters from `config/roasters.json` and denylisted domains from `config/denylist.txt`.
2. Fetch product lists and, when needed, product pages with robots compliance and retry/backoff.
3. Track first-seen products in SQLite and classify products as new within the current 7-day window.
4. Write per-roaster catalog JSON plus combined run-level JSON.
5. Use the `coffee-scout` skill in Codex to review top picks, skips, uncertainty, and follow-up questions.

## Code Structure

- `scrape_coffee.py` - explicit scrape entrypoint
- `main.py` - backward-compatible thin entrypoint
- `coffee_watch/cli.py` - CLI parsing and config/bootstrap error handling
- `coffee_watch/config.py` - settings model, CLI flags, config precedence, validation
- `coffee_watch/runner.py` - high-level scrape orchestration
- `coffee_watch/roaster_pipeline.py` - per-roaster scrape, classification, and catalog writing
- `coffee_watch/catalog.py` - normalized catalog schema helpers
- `coffee_watch/classify.py` - new-product date resolution
- `coffee_watch/network.py` - HTTP fetching, retries, robots, sitemaps
- `coffee_watch/http_limits.py` - per-host/global concurrency limiter
- `coffee_watch/report_status.py` - structured status sidecars for resume logic
- `coffee_watch/reporting.py` - JSON output helpers
- `coffee_watch/seen_products.py` - SQLite first-seen tracker
- `skills/coffee-scout/` - Codex skill for interactive coffee analysis
- `tests/` - pytest coverage

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Editable install with dev tooling:

```bash
pip install -e .[dev]
```

## Usage

Run with defaults:

```bash
python scrape_coffee.py
```

Show CLI help:

```bash
python scrape_coffee.py --help
```

Retry missing or failed roaster catalogs for the current UTC day:

```bash
python scrape_coffee.py --resume
```

Disable product-page fetches and use catalog payload text only:

```bash
python scrape_coffee.py --no-fetch-product-pages
```

Use an alternate config:

```bash
python scrape_coffee.py --config config/settings.json
```

## Configuration

Config precedence is:

```text
CLI flags > JSON config file > built-in defaults
```

Example `config/settings.json`:

```json
{
  "http_timeout_s": 20.0,
  "http_max_retries": 2,
  "jitter_min_s": 0.7,
  "jitter_max_s": 2.0,
  "http_concurrency": 1,
  "per_host_concurrency": 1,
  "sitemap_max_pages": 8,
  "max_products_per_source": 200,
  "page_text_max_chars": 0,
  "log_json_max_chars": 0,
  "fetch_product_pages": true,
  "resume": false,
  "save_pretty_products_json": false,
  "save_raw_products_json": false,
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

## Outputs

Generated local outputs include:

- `reports/YYYYMMDD-<roaster-slug>.catalog.json` - normalized per-roaster catalog
- `reports/YYYYMMDD-catalog.json` - combined run catalog
- `reports/YYYYMMDD-new-products.json` - flattened new-product subset
- `reports/YYYYMMDD-<roaster-slug>.status.json` - structured roaster scrape status
- `logs/assets/` - raw/pretty source payloads when enabled
- `logs/coffee_watch.log` - text or JSON log output
- `logs/seen_products.db` - SQLite seen-product store

Product entries include fields such as roaster, product URL, title, price, origin, process, tasting notes, availability, first-seen timestamp, raw product text, source metadata, and scrape errors. Origin/process/tasting-note extraction is best-effort and never replaces the source URL.

## Coffee Scout Skill

The repo includes `skills/coffee-scout`, a Codex skill for running the scraper and analyzing the fresh JSON. It uses preferences stated in the current conversation, so the first answer can be a broad recommendation and follow-up questions can refine it. The skill produces:

- top picks
- maybe list
- skip list
- what changed
- uncertainty notes
- follow-up questions

The skill treats scraped product text as untrusted evidence. It should not follow instructions embedded in roaster descriptions.

## Testing

Run tests:

```bash
pytest -q
```

Quick syntax check:

```bash
python -m py_compile scrape_coffee.py main.py coffee_watch/*.py tests/*.py
```

## Responsible Use

This is a hobby project intended for low-frequency monitoring and research.

Please use it responsibly:

- Review and follow each site's terms of service and `robots.txt`.
- Do not bypass paywalls, authentication, access controls, or anti-bot protections.
- If a site owner asks not to be monitored, add the domain to `config/denylist.txt`.

## License

MIT
