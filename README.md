# Coffee Watch (Portfolio Edition)

A low-frequency monitoring agent that checks specialty coffee roasters for new releases and evaluates them with Gemini + Google Search grounding. This repo is designed to showcase **production-grade MLE + backend engineering** with a strong emphasis on **ethical, transparent data collection**.

## Why this repo is safe to publish
- No secrets checked in; API keys are read from environment variables at runtime.
- Generated artifacts (logs, reports, local DB) are excluded via `.gitignore`.
- Network behavior is polite and transparent (strict User-Agent, robots.txt compliance, jittered requests).

## Key features
- **Polite network layer**: async `httpx` with HTTP/2, robots.txt checks, jittered pacing.
- **Transparent identity**: strict User-Agent string that points to this repo and a contact channel.
- **Low-frequency monitoring**: designed to run occasionally, not crawl aggressively.
- **Grounded evaluation**: Gemini with Google Search grounding for each roaster batch.
- **Structured logging**: rich logs for requests, prompts, grounding, and outcomes.
- **Fresh evaluation every run**: no dedupe/persistence yet (intentional for now).

## Code structure
- `main.py` is the thin entrypoint that calls `coffee_watch/cli.py`.
- `coffee_watch/runner.py` orchestrates the run lifecycle and report generation.
- `coffee_watch/network.py` handles HTTP, robots.txt, and jittered fetches.
- `coffee_watch/parsing.py` parses roaster configs and product lists.
- `coffee_watch/prompts.py` builds Gemini prompts and language helpers.
- `coffee_watch/gemini.py` wraps Gemini calls + grounding extraction.

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
python main.py --language zh --http-concurrency 4 --skip-gemini
python main.py --gemini-timeout-s 600
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
  "http_concurrency": 4,
  "http_timeout_s": 20.0,
  "jitter_min_s": 0.7,
  "jitter_max_s": 2.0,
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

## Configuration
- `config/roasters.json` controls sources, endpoints, and per-roaster settings.
- `config/denylist.txt` can block domains on request (one per line).

## Outputs (generated locally)
- `reports/` — Markdown reports + prompt captures
- `logs/coffee_watch.log` — request/response and Gemini traces

## Bot Policy & Opt-out
This bot runs at low volume and obeys robots.txt. If you are a site owner and prefer not to be monitored, please open a GitHub Issue, and we will add your domain to our denylist immediately.

## License
MIT
