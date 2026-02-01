# Coffee Watch

I'm a machine learning engineer who loves making and drinking coffee. I pursue top-quality and interesting coffee choices, but I don't have detailed knowledge about choosing roasters, farms, or beans. This is an agentic LLM system to help me make informed decisions.

A low-frequency monitoring agent that checks specialty coffee roasters for new releases and evaluates them with Gemini + Google Search grounding.

## Highlights
- Polite crawling with robots.txt checks, jittered pacing, and a fixed User-Agent.
- Batch evaluation with grounded Gemini outputs and saved markdown reports.
- Config-driven sources for easy customization.
- Structured logs for requests, prompts, and outcomes.
- SQLite page cache with conditional requests (ETag/Last-Modified) to avoid double pulls.
- Exponential retry with jitter on transient failures (e.g., 429/5xx).
- Stateless runs; outputs are written to `reports/` and `logs/`.

## How it works
1. Load roaster sources from `config/roasters.json`.
2. Fetch product lists and (optionally) product pages with robots.txt compliance.
3. Use cached page text when available; otherwise fetch with conditional requests.
4. Build a batch prompt from product metadata and sanitized page text.
5. Generate a per-roaster report, a full digest, and a new-products digest.

## Code structure
- `main.py` is the thin entrypoint that calls `coffee_watch/cli.py`.
- `coffee_watch/runner.py` orchestrates the run lifecycle and report generation.
- `coffee_watch/network.py` handles HTTP, robots.txt, and jittered fetches.
- `coffee_watch/parsing.py` parses roaster configs and product lists.
- `coffee_watch/catalog_parsers.py` handles site-specific catalog parsing (e.g., Wix).
- `coffee_watch/prompts.py` builds Gemini prompts and language helpers.
- `coffee_watch/gemini.py` wraps Gemini calls + grounding extraction.
- `coffee_watch/page_cache.py` stores cached page text + headers in SQLite.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export GEMINI_API_KEY=your_key_here
python main.py
```

### CLI usage
Run with defaults:
```bash
python main.py
```

Override on the CLI:
```bash
python main.py --language zh --http-concurrency 1 --skip-gemini
python main.py --gemini-timeout-s 600
python main.py --cache-max-age-s 0  # always revalidate
```

### Testing
Basic sanity check:
```bash
python -m py_compile main.py coffee_watch/*.py
```

### Config file overrides
Pass a JSON config file and override selectively with CLI flags. CLI > config > defaults.
```bash
python main.py --config config/settings.json --language en
```

Example `config/settings.json`:
```json
{
  "language": "zh",
  "model": "gemini-3-pro-preview",
  "gemini_timeout_s": 600.0,
  "http_concurrency": 1,
  "http_timeout_s": 20.0,
  "jitter_min_s": 1.5,
  "jitter_max_s": 3.0,
  "max_products_per_source": 200,
  "page_text_max_chars": 0,
  "batch_page_text_max_chars": 0,
  "log_json_max_chars": 0,
  "fetch_only": false,
  "skip_gemini": false,
  "save_prompt": false,
  "save_pretty_products_json": false,
  "save_raw_products_json": false,
  "save_report": true,
  "cache_db_path": "logs/coffee_watch_cache.sqlite",
  "cache_max_age_s": 21600.0,
  "roasters_path": "config/roasters.json",
  "denylist_path": "config/denylist.txt",
  "reports_dir": "reports",
  "log_path": "logs/coffee_watch.log",
  "log_level": "INFO"
}
```

Notes:
- Only `GEMINI_API_KEY` is read from the environment.
- Descriptions are extracted from product `body_html` (when available).
- `gemini_timeout_s` controls Gemini request timeouts in seconds (0 = no timeout).
- Reports are saved as `YYYYMMDD-roaster-slug.md` and `YYYYMMDD-digest.md` (UTC date).
- A new-products digest is saved as `YYYYMMDD-new-digest.md` when new coffees are detected.
- Cached page text is stored in `logs/coffee_watch_cache.sqlite`.

## Configuration
- `config/roasters.json` controls sources, endpoints, and per-roaster settings.
- `config/denylist.txt` can block domains on request (one per line).

## Outputs (generated locally)
- `reports/` — Markdown reports + prompt captures
- `logs/coffee_watch.log` — request/response and Gemini traces
- `logs/coffee_watch_cache.sqlite` — cached page text + last-modified/etag metadata

## Support & Opt-out
For questions or issues, open a GitHub Issue.

This bot runs at low volume and obeys robots.txt. If you are a site owner and prefer not to be monitored, please open a GitHub Issue, and we will add your domain to our denylist immediately.

## Responsible use
This is a hobby project intended for low-frequency monitoring and research.

Please use it responsibly:
- Review and follow each site's terms of service and robots.txt.
- Do not bypass access controls, paywalls, authentication, or anti-bot measures.
- You are responsible for how you use this tool; the author is not responsible for misuse.

## License
MIT
