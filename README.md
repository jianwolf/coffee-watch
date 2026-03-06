# Coffee Watch

I'm a machine learning engineer who loves making and drinking coffee. I pursue top-quality and interesting coffee choices, but I don't have detailed knowledge about choosing roasters, farms, or beans. This is an agentic LLM system to help me make informed decisions.

A low-frequency monitoring agent that checks specialty coffee roasters for new releases and evaluates them with either Gemini + Google Search grounding or a local MLX-served model such as Qwen.

## Highlights
- Polite crawling with robots.txt checks, jittered pacing, and a fixed User-Agent.
- Batch evaluation with grounded Gemini outputs or local MLX outputs, with saved markdown reports.
- Config-driven sources for easy customization.
- Structured logs for requests, prompts, and outcomes.
- SQLite seen-products store to track first-seen items by URL/title/description hash.
- Exponential retry with jitter on transient failures (e.g., 429/5xx).
- Per-roaster LLM retry up to 10 attempts; hard failures are explicitly recorded.
- Stateless runs; outputs are written to `reports/` and `logs/`.

## Prompting philosophy
- Roaster and digest task prompts are intentionally shared across Gemini and local/open-source models.
- This is a deliberate learning goal: keep instructions constant, then compare model quality, runtime behavior, quantization, and serving choices without prompt drift contaminating the comparison.
- Backend-specific differences should stay in transport/runtime configuration, not in the task wording, unless there is a clear product requirement that outweighs comparability.

## How it works
1. Load roaster sources from `config/roasters.json`.
2. Fetch product lists and (optionally) product pages with robots.txt compliance.
3. Track first-seen products in SQLite and classify new items by publish date, page headers, or first-seen timestamp.
4. Build a batch prompt from product metadata and sanitized page text.
5. Generate a per-roaster report, a full digest, a new-products digest (past 7 days), and a roaster ratings digest.
6. If any roaster report failed, append a `## Report Generation Failures` section to digest outputs.

## Code structure
- `main.py` is the thin entrypoint that calls `coffee_watch/cli.py`.
- `coffee_watch/runner.py` orchestrates the run lifecycle and report generation.
- `coffee_watch/network.py` handles HTTP, robots.txt, and jittered fetches.
- `coffee_watch/parsing.py` parses roaster configs and product lists.
- `coffee_watch/catalog_parsers.py` handles site-specific catalog parsing (e.g., Wix).
- `coffee_watch/prompts.py` builds report prompts and language helpers.
- `coffee_watch/gemini.py` wraps Gemini calls + grounding extraction.
- `coffee_watch/llm.py` selects the active backend and bridges Gemini vs local MLX.
- `coffee_watch/seen_products.py` stores first-seen products in SQLite.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py
```

Gemini override:
```bash
export GEMINI_API_KEY=your_key_here
python main.py --llm-backend gemini
```

### CLI usage
Run with defaults:
```bash
python main.py
```

Override on the CLI:
```bash
python main.py --language zh --http-concurrency 1 --skip-llm
python main.py --llm-timeout-s 600
python main.py --seen-db-path logs/seen_products.db
python main.py --digest-only        # regenerate digests from today's roaster reports (UTC)
python main.py --resume             # retry only missing/failed reports today, then rebuild digests
python main.py --llm-backend mlx --mlx-runtime vlm --mlx-model mlx-community/Qwen3.5-9B-MLX-8bit
python main.py --llm-backend mlx --mlx-runtime vlm --mlx-trust-remote-code
python main.py --stream-llm-output
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
  "llm_backend": "mlx",
  "model": "mlx-community/Qwen3.5-9B-MLX-8bit",
  "digest_model": "mlx-community/Qwen3.5-9B-MLX-8bit",
  "gemini_timeout_s": 600.0,
  "mlx_model": "mlx-community/Qwen3.5-9B-MLX-8bit",
  "mlx_runtime": "vlm",
  "mlx_host": "127.0.0.1",
  "mlx_port": 8080,
  "mlx_startup_timeout_s": 900.0,
  "mlx_trust_remote_code": false,
  "http_concurrency": 1,
  "http_timeout_s": 20.0,
  "jitter_min_s": 0.7,
  "jitter_max_s": 2.0,
  "max_products_per_source": 200,
  "page_text_max_chars": 0,
  "batch_page_text_max_chars": 0,
  "log_json_max_chars": 0,
  "fetch_only": false,
  "skip_gemini": false,
  "stream_llm_output": true,
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
  "log_path": "logs/coffee_watch.log",
  "log_level": "INFO"
}
```

Notes:
- Plain `python main.py` now defaults to the local MLX backend.
- `GEMINI_API_KEY` is only required when `llm_backend` is `gemini`.
- Prompt text is intentionally shared between Gemini and local MLX runs so report quality differences are easier to attribute to the model/runtime rather than prompt differences. The local backend does not inject extra task instructions beyond transport/runtime settings.
- The current comparison baseline also uses the same decoding temperature on both backends: `1.0` for Gemini and local MLX. This was chosen to stay closer to vendor-recommended starting points and because a lower shared temperature (`0.2`) was observed to reinforce repetitive self-check loops on the small local Qwen model.
- Descriptions are extracted from product `body_html` (when available).
- Shopify sources rely on `products.json` and skip per-product page fetches.
- `gemini_timeout_s` / `llm_timeout_s` controls LLM request timeouts in seconds (0 = no timeout).
- `model` applies to per-roaster reports; `digest_model` applies to digest generation.
- `llm_backend=mlx` starts or reuses a local MLX server at a runtime-specific base URL: `http://<mlx_host>:<mlx_port>/v1` for `mlx_runtime=lm`, and `http://<mlx_host>:<mlx_port>` for `mlx_runtime=vlm`.
- `mlx_runtime=vlm` starts `mlx_vlm.server`; `mlx_runtime=lm` starts `mlx_lm.server`.
- `mlx_trust_remote_code` passes `--trust-remote-code` to the MLX server when needed by a model.
- `mlx_model` defaults to `mlx-community/Qwen3.5-9B-MLX-8bit`, and is also used as the default `model` and `digest_model` when `llm_backend=mlx`.
- `stream_llm_output` mirrors streamed MLX output to the terminal on the local backend. It now defaults to `true`; use `--no-stream-llm-output` to disable it.
- `new_products_digest` toggles the new-products digest report (default `true`).
- Per-roaster LLM generation retries up to 10 times before writing a hard-failure note.
- `resume` retries only missing/failed roaster reports for today (UTC), then regenerates digests from all today reports.
- The new-products digest includes coffees discovered in the last 7 days (inclusive, UTC run day).
- `--digest-only` regenerates digests from today's existing roaster reports and does not update `logs/seen_products.db`.
- Digest outputs append `## Report Generation Failures` with lines like `xxx roaster report generation has failed` when any roaster report is missing/failed.
- Reports are saved as `YYYYMMDD-roaster-slug.md`, `YYYYMMDD-z-digest.md`, and `YYYYMMDD-z-roaster-digest.md` (UTC date).
- A new-products digest is saved as `YYYYMMDD-z-new-digest.md` when enabled and matching coffees are detected.
- Seen products are stored in `logs/seen_products.db`.

## Configuration
- `config/roasters.json` controls sources, endpoints, and per-roaster settings (including `platform`).
- `config/denylist.txt` can block domains on request (one per line).

## Outputs (generated locally)
- `reports/` — Markdown reports + prompt captures
- `logs/coffee_watch.log` — request/response and LLM traces
- `logs/seen_products.db` — seen product hashes and first-seen timestamps

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
