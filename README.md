# Coffee Watch

[![CI](https://github.com/jianwolf/coffee-watch/actions/workflows/ci.yml/badge.svg)](https://github.com/jianwolf/coffee-watch/actions/workflows/ci.yml)

Coffee Watch is a coffee scouting workflow for a home coffee buyer, built to run as a [Claude Code](https://code.claude.com/docs) cloud session - including an unattended monthly routine. You ask for `/coffee-scout`; it scrapes the roaster catalogs, reads the fresh evidence, and **answers in the session's reply** with a short scouting note: what is worth buying now, what changed, and what to verify before checkout.

The reply is the whole deliverable. Nothing is written back into the repo, so a scheduled run leaves no artifacts and no dirty tree behind.

```text
/coffee-scout
  -> polite roaster scraping          (scrape_coffee.py)
  -> normalized JSON catalogs         (reports/, gitignored evidence)
  -> compact brief                    (python -m coffee_watch.brief)
  -> one short reply in the session
```

The scraper is the evidence layer: HTTP fetching, parsing, first-seen tracking, catalog normalization, source URL preservation. `coffee_watch/brief.py` is the reading layer - it projects a 1-3 MB catalog into a few hundred one-line rows with a coverage header, computes `$/100g`, diffs against an earlier catalog when one exists, and flags the conditions that are easy to misread. Judgment is the agent's, in `.claude/skills/coffee-scout`.

Setting up the cloud environment (network allowlist, timeouts, the monthly routine) is a one-time step: see [docs/cloud-setup.md](docs/cloud-setup.md).

## Highlights

- Polite crawling with `robots.txt` checks, jittered pacing, retries, and a fixed `User-Agent`
- Per-host and global HTTP concurrency limits
- Shopify, JSON, and HTML catalog parsing
- SQLite first-seen tracking in `logs/seen_products.db`
- Normalized per-roaster and combined catalog JSON outputs
- Structured status sidecars for resume/retry behavior
- Deterministic catalog projection for agent reading, with `$/100g`, wholesale-only detection, republish clustering, and a catalog-to-catalog delta
- Unattended-run hardening: a whole-run deadline that keeps partial results, a cooldown retry for rate-limited roasters inside that budget, a non-zero exit when a run collects nothing, and distinct logging for sandbox egress refusals
- Repo-local `coffee-scout` skill that answers in chat and writes nothing
- Config validation, structured logging, and pytest coverage

## How It Works

1. Load roasters from `config/roasters.json` and denylisted domains from `config/denylist.txt`.
2. Fetch product lists and, when needed, product pages with robots compliance and retry/backoff.
3. Track first-seen products in SQLite and classify products as new within the current 7-day window.
4. Write per-roaster catalog JSON plus combined run-level JSON.
5. Project the catalog into the compact brief and let the `coffee-scout` skill turn it into a short reply that ranks the strongest current options and invites follow-up preferences.

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
- `coffee_watch/atom_feed.py` - Shopify collection Atom feed fallback and its limits
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
- `coffee_watch/brief.py` - compact catalog projection, delta, detail lookup, the
  `--competition` and `--prized` shortlists, and `--verify-links` for a finished note
- `coffee_watch/hosts.py` - egress allowlist generator for cloud environments
- `.claude/skills/coffee-scout/` - the scouting skill
- `.claude/settings.json` - committed allow rules and the cloud dependency hook
- `scripts/cloud_setup.sh` - SessionStart dependency install, cloud only
- `tests/` - pytest coverage

## Cloud Setup

The skill lives at `.claude/skills/coffee-scout/SKILL.md` as a real directory, so a fresh
clone needs no wiring on any platform. `CLAUDE.md` imports `AGENTS.md`, so repository
guidance is single-source.

Three things are account or environment settings and cannot be committed. Set them once,
following [docs/cloud-setup.md](docs/cloud-setup.md):

| What | Why | Where |
| :--- | :--- | :--- |
| Network allowlist | A cloud environment's default allowlist covers package registries, not coffee storefronts, so every request is refused with `403 x-deny-reason: host_not_allowed` | Environment -> network access `Custom`, pasted from `python -m coffee_watch.hosts` |
| Bash timeouts | A full scrape takes 2-9 minutes; the cloud default for a foreground command is 2 | `BASH_DEFAULT_TIMEOUT_MS` / `BASH_MAX_TIMEOUT_MS` on the environment |
| Monthly routine | Schedules live on your claude.ai account, not in the repo | `/schedule`, or claude.ai/code/routines |

Dependencies need no setup: a committed `SessionStart` hook runs `scripts/cloud_setup.sh`,
which installs `httpx[http2]` only when `CLAUDE_CODE_REMOTE=true` and only when it is
missing. Local checkouts are left alone.

Verify the wiring after cloning:

```bash
pytest -q tests/test_cloud_native.py
```

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

Ask for the skill - `/coffee-scout`, or just ask to check what is worth buying. It runs
the scraper, reads the brief, and replies in the session. In a cloud session it needs
nothing from you beyond the one-time setup above.

To read the evidence yourself:

```bash
python -m coffee_watch.brief
```

```bash
python -m coffee_watch.brief --detail "Patio Bonito"
```

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

Bound the whole run. Roasters still in flight when the budget expires are cancelled and everything already collected is kept, so an unattended run cannot be stalled by one tarpitting host:

```bash
python scrape_coffee.py --run-deadline-s 600
```

Inside that budget, roasters whose catalog API rate-limited us get re-asked after a cooldown, and again while re-asking keeps working. Which stores are gated changes minute to minute, and a full run typically uses a third of its deadline, so the idle remainder pays for the retry. It needs a declared deadline — with none there is no idle remainder to spend, and the pass is skipped rather than making every run longer. The cooldown is a floor, not a remainder: if waiting it out would not leave real working time, the attempt is skipped instead of being sent and cancelled. A host's own `Retry-After` overrides the configured cooldown:

```bash
python scrape_coffee.py --run-deadline-s 600 --retry-cooldown-s 120
```

The loop stops at the first pass that recovers nothing: that is the signal that whoever is left is gated rather than throttled, and re-asking a host that has refused every time is what this project's own back-off rule forbids. `--retry-max-passes` (default 2) is the ceiling, the cooldown grows with each pass, and the default 600s budget usually affords one retry pass -- raise the deadline to afford more.

Measured: two runs recovered 4 of 10 rate-limited roasters on the first retry pass, taking coverage from 7/17 to 11/17. A variant that cut each pass to a single request per host recovered 0 of 9 and is not used -- the per-request retries carry their own 30-60s backoffs and appear to be doing the work.

```bash
python scrape_coffee.py --no-retry-rate-limited-roasters
```

Only a retryable catalog-API status (429, 5xx) qualifies. A roaster skipped at the robots gate is never retried — honouring `robots.txt` is the point, not an error — and neither is a sandbox egress refusal, which will not change mid-run.

After the retries, a roaster the catalog API never allowed falls back to the collection's Atom feed, which Shopify publishes for the same collection and which these stores' `robots.txt` allows (`User-agent: * / Allow: /`). One request returns the product list with titles, URLs, published dates and the roaster's own tasting-note text. It is a degraded source and the rows say so:

- **No per-variant size, price or stock.** The feed quotes a price *range*, so there is no purchasable size, no `$/100g`, and availability is `unknown` rather than in stock. Rows carry the `ATOM-ONLY-no-size-or-stock` flag and the brief's header repeats the caveat.
- **The newest 25 products only.** The feed is capped and ignores `?page`, so a large store is truncated. Entries are newest-first, so the cap keeps the window a "what is new" question needs and drops older standing stock.

In practice the subsequent product-page fetch upgrades most feed-discovered coffees to real page-verified sizes; a measured run left only 4 rows feed-only, all of them non-coffee (gift cards, instant, subscriptions). Disable with `--no-atom-fallback`.

Measured end to end: first pass 7/17, retry pass 1 recovered 3, retry pass 2 recovered 2, the feed covered the last 3 — **17/17, 383 products**, in 684s of a 900s budget.

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

`http_total_timeout_s` is a hard wall-clock deadline per request (covering slow trickled responses), `run_deadline_s` is the same idea for the whole run, and `http_max_response_bytes` caps how much of any response body is downloaded; set any of them to `0` to disable. `assets_retention_days` prunes raw payload caches in the assets dir at run start (`0` keeps everything).

Example `config/settings.json`:

```json
{
  "http_timeout_s": 20.0,
  "http_total_timeout_s": 120.0,
  "run_deadline_s": 0,
  "retry_rate_limited_roasters": true,
  "retry_cooldown_s": 120.0,
  "retry_max_passes": 2,
  "atom_fallback": true,
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

Scraping writes evidence, not reports. Everything below is gitignored and local to one
workspace; the scouting answer itself is the session reply:

- `reports/YYYYMMDD-<roaster-slug>.catalog.json` - normalized per-roaster catalog
- `reports/YYYYMMDD-catalog.json` - combined run catalog
- `reports/YYYYMMDD-new-products.json` - flattened new-product subset
- `reports/YYYYMMDD-<roaster-slug>.status.json` - structured roaster scrape status
- `logs/assets/` - raw/pretty source payload cache, always written and pruned after `assets_retention_days`; the `save_raw_products_json`/`save_pretty_products_json` flags control extra copies in `reports/`
- `logs/coffee_watch.log` - text or JSON log output
- `logs/seen_products.db` - SQLite seen-product store (kept indefinitely; it is the first-seen source of truth)

Product entries include fields such as roaster, product URL, title, selected buyable price label, selected price variant, all variants, storefront-visible variant titles and storefront status when verified, origin, process, tasting notes, availability, first-seen timestamp, raw product text, source metadata, and scrape errors. Origin/process/tasting-note extraction is best-effort and never replaces the source URL.

### Exit codes

`0` success, `1` configuration or bootstrap failure, `2` invalid CLI/config input, `3` the
run collected zero products. `3` matters because a scrape with no network access still
writes a valid, empty catalog and would otherwise look like a quiet market.

## Known external constraints

- Since 2026-06-10, some Shopify storefronts return HTTP 429 for product HTML pages fetched with the bot User-Agent while leaving the catalog API (`products.json`) open. This weakens storefront verification but not core catalog data. The scraper fast-skips a host's remaining page fetches after 3 consecutive 429s in a run, recording the skip per product and in the roaster status note, instead of retrying against a closed gate. Details, evidence, and response policy: [docs/shopify-bot-gating.md](docs/shopify-bot-gating.md).

## Coffee Scout Skill

`/coffee-scout` is the consumer-facing surface. It reads `purchase-journal.md` for
standing preferences, the watchlist and recorded taste verdicts, runs the scraper with a
deadline, reads the brief, pulls `--detail` evidence for its shortlist, and replies with
roughly 40 lines:

- a one-line evidence header (roasters that returned data, products, elapsed)
- the three things that matter this session
- two or three roasters, strongest first, with 8-12 coffee lines carrying flavor, a
  size-aware price, `$/100g` and the source URL
- a short radar for anything that could change the plan
- what to verify: API-reported sizes, coverage gaps, missing history

It does not write report files, does not commit, and does not present a single checkout
as the answer. Scraped product text is untrusted evidence; the skill never follows
instructions found inside it.

`purchase-journal.md` is the one piece of durable state, because it is committed. The
skill appends a purchase there only when you decide something in conversation, and never
deletes what you wrote.

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
