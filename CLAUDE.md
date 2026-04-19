# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run (defaults to Gemini backend)
python main.py

# Run with Gemini backend
export GEMINI_API_KEY=your_key_here
python main.py --llm-backend gemini --model gemini-3.1-pro-preview --digest-model gemini-3.1-pro-preview

# Common flag combos
python main.py --digest-only               # regenerate digests from today's reports only
python main.py --resume                    # retry failed/missing reports then rebuild digests
python main.py --skip-llm                  # fetch and parse only, no LLM calls
python main.py --ask "I want decaf"        # personalize recommendations (max one --ask)
python main.py --config config/settings.json --language en

# Syntax check (no formal test suite)
python -m py_compile main.py coffee_watch/*.py
```

## Architecture

**Entry point:** `main.py` → `coffee_watch/cli.py` → `coffee_watch/runner.py`

**Config precedence:** CLI flags > `config/settings.json` > built-in defaults (defined in `coffee_watch/config.py` as a `Settings` dataclass).

**Run lifecycle** (`runner.py`):
1. Load roasters from `config/roasters.json`, filter by `config/denylist.txt`
2. Fetch product lists (JSON for Shopify, HTML for Wix) via `network.py` with robots.txt compliance and jittered pacing
3. Deduplicate via SHA3-512 hashes in `logs/seen_products.db` (`seen_products.py`)
4. Build batch prompts per roaster (`prompts.py`) and call LLM (`llm.py`)
5. Write per-roaster markdown reports, then three digest reports to `reports/`

**LLM backends** (`llm.py` routes to one of):
- `gemini.py` — Google Gemini API with optional Search grounding
- `mlx_server.py` — auto-starts a local `mlx_lm.server` or `mlx_vlm.server` subprocess and talks to it over HTTP

**Prompt policy:** Task prompts in `prompts.py` are intentionally identical across backends. This is a deliberate design goal so that report quality differences are attributable to the model/runtime rather than prompt wording. Backend-specific logic belongs in transport/runtime code, not prompt text.

**Output files** (all UTC date-stamped):
- `reports/YYYYMMDD-<roaster-slug>.md` — per-roaster evaluation
- `reports/YYYYMMDD-z-digest.md` — full digest
- `reports/YYYYMMDD-z-roaster-digest.md` — roaster ratings digest
- `reports/YYYYMMDD-z-new-digest.md` — new products (last 7 days), when enabled
- `logs/coffee_watch.log` — structured JSON logs
- `logs/seen_products.db` — SQLite seen-product hashes

## Key Design Decisions

- **Concurrency:** default `http_concurrency=1` for polite crawling; jitter `0.7–2.0s` between requests.
- **Retries:** per-roaster LLM calls retry up to 10 times; HTTP uses exponential backoff on 429/5xx.
- **Temperature:** `1.0` on both backends — a lower shared value (`0.2`) was found to cause repetitive self-check loops on small local Qwen models.
- **MLX backend URL:** `lm` runtime uses `/v1` base path; `vlm` runtime uses root `/`.
- **`--digest-only`** does not update `seen_products.db`; `--resume` does.
- Failures are surfaced as a `## Report Generation Failures` section appended to digest outputs rather than hard-crashing.
