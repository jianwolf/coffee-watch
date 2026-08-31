# Coffee Watch

[![CI](https://github.com/jianwolf/coffee-watch/actions/workflows/ci.yml/badge.svg)](https://github.com/jianwolf/coffee-watch/actions/workflows/ci.yml)

Coffee Watch is an agent-operated coffee scouting workflow for a home coffee buyer. Your coding agent runs the bundled `coffee-scout` skill, which scrapes roaster catalogs, writes normalized JSON evidence, saves markdown digests, and produces an interactive Chinese buying report that ranks standout coffees by roaster for follow-up discussion. The report is a scouting menu, not a final checkout decision.

The skill is agent-independent: the same `SKILL.md` runs in [Codex](https://developers.openai.com/codex), [Claude Code](https://code.claude.com/docs), and [Cursor](https://cursor.com/docs) with no per-agent variant. See [Agent Setup](#agent-setup).

Current workflow:

```text
Coffee Scout skill (any agent)
  -> polite roaster scraping
  -> normalized JSON catalogs and status files
  -> markdown digest reports
  -> interactive buying recommendations
```

The scraper is the evidence-collection layer: HTTP fetching, parsing, first-seen tracking, catalog normalization, and source URL preservation. Coffee analysis and purchase reasoning happen in the agent through `.agents/skills/coffee-scout`.

## Highlights

- Polite crawling with `robots.txt` checks, jittered pacing, retries, and a fixed `User-Agent`
- Per-host and global HTTP concurrency limits
- Shopify, JSON, and HTML catalog parsing
- SQLite first-seen tracking in `logs/seen_products.db`
- Normalized per-roaster and combined catalog JSON outputs
- Structured status sidecars for resume/retry behavior
- Repo-local `coffee-scout` skill for interactive buying analysis, shared by Codex, Claude Code, and Cursor
- Agent-written Chinese markdown digests and a final purchase report under `reports/`
- Config validation, structured logging, and pytest coverage

## How It Works

1. Load roasters from `config/roasters.json` and denylisted domains from `config/denylist.txt`.
2. Fetch product lists and, when needed, product pages with robots compliance and retry/backoff.
3. Track first-seen products in SQLite and classify products as new within the current 7-day window.
4. Write per-roaster catalog JSON plus combined run-level JSON.
5. Use the `coffee-scout` skill in your agent to write the three digest markdown files plus a final report that ranks standout coffees by roaster and invites follow-up preferences.

## Code Structure

- `scrape_coffee.py` - explicit scrape entrypoint
- `main.py` - backward-compatible thin entrypoint
- `coffee_watch/cli.py` - CLI parsing and config/bootstrap error handling
- `coffee_watch/config.py` - settings model, CLI flags, config precedence, validation
- `coffee_watch/context.py` - run-scoped context bundle shared across the pipeline
- `coffee_watch/models.py` - roaster/product/variant/status dataclasses
- `coffee_watch/runner.py` - high-level scrape orchestration
- `coffee_watch/roaster_pipeline.py` - per-roaster scrape, classification, and catalog writing
- `coffee_watch/catalog.py` - normalized catalog schema helpers
- `coffee_watch/catalog_parsers.py` - named catalog HTML parsers (e.g. Wix shop pages)
- `coffee_watch/parsing.py` - roaster config loading and product payload parsing
- `coffee_watch/classify.py` - new-product date resolution
- `coffee_watch/network.py` - HTTP fetching, retries, robots, sitemaps
- `coffee_watch/http_limits.py` - per-host/global concurrency limiter and 429 fast-skip gate
- `coffee_watch/text_utils.py` - HTML-to-text, price/size extraction, sanitization
- `coffee_watch/url_utils.py` - URL normalization, pattern matching, denylist checks
- `coffee_watch/report_status.py` - structured status sidecars for resume logic
- `coffee_watch/reporting.py` - JSON output helpers and assets retention pruning
- `coffee_watch/logging_utils.py` - text/JSON logging setup
- `coffee_watch/seen_products.py` - SQLite first-seen tracker
- `.agents/skills/coffee-scout/` - cross-vendor agent skill for interactive coffee analysis
- `tests/` - pytest coverage

## Agent Setup

The skill body lives once, at `.agents/skills/coffee-scout/SKILL.md`. Each agent picks it
up from a location it already scans, so a fresh clone needs no configuration:

| Agent | Reads the skill from | Reads repo guidance from |
| :--- | :--- | :--- |
| Codex | `.agents/skills/` (scanned to the repo root) | `AGENTS.md` |
| Cursor | `.agents/skills/` (also supports `.cursor/skills/`) | `AGENTS.md` |
| Claude Code | `.claude/skills/coffee-scout` -> symlink to `.agents/skills/coffee-scout` | `CLAUDE.md`, which imports `AGENTS.md` |

Nothing in the skill depends on a particular agent, model, or vendor tool. Model choice is
yours - Codex with GPT models, Claude Code with Claude models, Cursor with Grok 4.6 or
anything else it offers. Optional features such as subagents are used when available and
skipped inline when they are not.

Verify the wiring after cloning:

```bash
pytest -q tests/test_agent_portability.py
```

On Windows, git needs symlink support for the Claude Code entry point. Either run
`git config --global core.symlinks true` before cloning, or replace
`.claude/skills/coffee-scout` with a copy of `.agents/skills/coffee-scout`.

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

## Normal Usage

For the intended workflow, ask your agent to use `coffee-scout` (or invoke it directly: `$coffee-scout` in Codex and Cursor, `/coffee-scout` in Claude Code). The agent should run the scraper, request network approval if the sandbox blocks DNS/HTTP access, write the digest markdown files in `reports/`, and then present the interactive Chinese buying report.

## Manual Scraper Usage

These commands are mainly for development, debugging, and manual verification. They are not the normal consumer-facing flow.

Run with defaults:

```bash
python scrape_coffee.py
```

If installed in editable mode, the equivalent console scripts are also available:

```bash
coffee-watch
scrape-coffee
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

Widen the new-product window (default 7 days) to match a longer buying cadence:

```bash
python scrape_coffee.py --new-window-days 30
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

Config files are validated before a run starts. Unknown keys, invalid numeric values, invalid boolean strings, and unsupported log levels fail fast with a configuration error. A missing or malformed `--config` file also aborts the run instead of silently falling back to defaults.

`http_total_timeout_s` is a hard wall-clock deadline per request (covering slow trickled responses), and `http_max_response_bytes` caps how much of any response body is downloaded; set either to `0` to disable. `assets_retention_days` prunes raw payload caches in the assets dir at run start (`0` keeps everything).

Example `config/settings.json`:

```json
{
  "http_timeout_s": 20.0,
  "http_total_timeout_s": 120.0,
  "http_max_response_bytes": 10000000,
  "http_max_retries": 2,
  "jitter_min_s": 0.7,
  "jitter_max_s": 2.0,
  "http_concurrency": 50,
  "per_host_concurrency": 1,
  "sitemap_max_pages": 8,
  "max_products_per_source": 200,
  "new_window_days": 7,
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
  "assets_retention_days": 30,
  "log_path": "logs/coffee_watch.log",
  "log_level": "INFO",
  "log_format": "text"
}
```

Per-roaster configs can set `verify_variant_pages: true` when a storefront hides sample, wholesale, or business-only variants from the product page even though they still appear in the catalog API. When enabled, Coffee Watch fetches product pages for that roaster and uses storefront-visible size choices, page price, and page text to choose the displayed buyable price label. For non-Shopify HTML/Wix products without API variants, visible product-page price plus Bag Size text can be promoted into the selected price label, such as `100g = $24.00`. Public product pages returning 401, 403, 404, or 410 are treated as not publicly buyable, so their API variants are preserved as evidence but not used for displayed pricing.

## Outputs

Generated local outputs include:

- `reports/YYYYMMDD-<roaster-slug>.catalog.json` - normalized per-roaster catalog
- `reports/YYYYMMDD-catalog.json` - combined run catalog
- `reports/YYYYMMDD-new-products.json` - flattened new-product subset
- `reports/YYYYMMDD-<roaster-slug>.status.json` - structured roaster scrape status
- `reports/YYYYMMDD-z-digest.md` - all-roaster digest
- `reports/YYYYMMDD-z-roaster-digest.md` - roaster scorecard digest
- `reports/YYYYMMDD-z-new-digest.md` - new-product digest
- `reports/YYYYMMDD-z-purchase-report.md` - final interactive purchase report
- `logs/assets/` - raw/pretty source payload cache, always written and pruned after `assets_retention_days`; the `save_raw_products_json`/`save_pretty_products_json` flags control extra copies in `reports/`
- `logs/coffee_watch.log` - text or JSON log output
- `logs/seen_products.db` - SQLite seen-product store (kept indefinitely; it is the first-seen source of truth)

Product entries include fields such as roaster, product URL, title, selected buyable price label, selected price variant, all variants, storefront-visible variant titles and storefront status when verified, origin, process, tasting notes, availability, first-seen timestamp, raw product text, source metadata, and scrape errors. Origin/process/tasting-note extraction is best-effort and never replaces the source URL.

## Known external constraints

- Since 2026-06-10, some Shopify storefronts return HTTP 429 for product HTML pages fetched with the bot User-Agent while leaving the catalog API (`products.json`) open. This weakens storefront verification but not core catalog data. The scraper fast-skips a host's remaining page fetches after 3 consecutive 429s in a run, recording the skip per product and in the roaster status note, instead of retrying against a closed gate. Details, evidence, and response policy: [docs/shopify-bot-gating.md](docs/shopify-bot-gating.md).

## Coffee Scout Skill

The repo includes `coffee-scout`, a skill for running the scraper and analyzing the fresh JSON. This system is intentionally designed for an agent to operate end to end: the agent runs `python scrape_coffee.py`, handles sandbox/network approval if needed, reads the fresh catalog JSON, writes markdown digest artifacts, and then presents an interactive Chinese buying report.

The skill uses preferences stated in the current conversation, so the first answer can be a focused shortlist and follow-up questions can refine it. The practical consumer context is a home purchase that may end with one roaster and about two bags, but the report should first give enough ranked menu information for the user to steer the next question without forcing every roaster into a two-bag bundle or overwhelming the user with every plausible option.

The skill produces four report surfaces:

- all-roaster digest
- roaster scorecard digest
- new-product digest
- final purchase report with a focused roaster shortlist, ranked coffee highlights, near misses, and preference-based narrowing suggestions

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
- Keep `per_host_concurrency` conservative by default; `http_concurrency` only
  removes unnecessary serialization across different hosts.
- Do not bypass paywalls, authentication, access controls, or anti-bot protections.
- If a site owner asks not to be monitored, add the domain to `config/denylist.txt`.

## License

MIT
