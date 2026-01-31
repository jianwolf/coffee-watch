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

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export GEMINI_API_KEY=your_key_here
python main.py
```

### Optional environment variables
- `COFFEEWATCH_LANGUAGE` = `en` (default) or `zh` for 简体中文 output
- `COFFEEWATCH_MODEL` (default: `gemini-3-pro-preview`)
- `COFFEEWATCH_HTTP_CONCURRENCY` (default: `4`)
- `COFFEEWATCH_SKIP_GEMINI=true` to only build prompts
- `COFFEEWATCH_FETCH_ONLY=true` to only fetch products
- `COFFEEWATCH_REPORTS_DIR`, `COFFEEWATCH_LOG_PATH` for paths

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
