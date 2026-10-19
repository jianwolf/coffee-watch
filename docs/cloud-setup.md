# Running Coffee Watch in a cloud session

Coffee Watch is built to run as a Claude Code cloud session: a fresh clone in a
disposable VM, scraping the roasters and answering in the session's reply. Three
things cannot live in the repo, because they are account and environment
settings. They are one-time setup.

## 1. Let the environment reach the roasters

A cloud environment defaults to the **Trusted** network level, whose allowlist
covers package registries and GitHub - not coffee storefronts. On that default
every roaster request is refused by the egress proxy with `403` and
`x-deny-reason: host_not_allowed`, and the run writes an empty catalog.

Create (or edit) an environment at claude.ai/code, set network access to
**Custom**, keep "include default list of common package managers" checked, and
paste the host list:

```bash
python -m coffee_watch.hosts
```

That prints every host the scrape needs: the configured `base_url` hosts, the
hosts the last catalog's product URLs actually resolved to (one roaster is
configured on a `myshopify.com` entry point that redirects elsewhere), and an
apex wildcard where the host is the roaster's own domain. Re-run it after
changing `config/roasters.json` and update the environment.

`Full` access also works and needs no maintenance; `Custom` is preferred because
it keeps the blast radius of a scraper bug to coffee shops.

## 2. Give the scrape enough time

A full 17-roaster scrape takes 2-9 minutes. A foreground command in a cloud
session gets 2 minutes by default. Set these environment variables on the
environment:

```text
BASH_DEFAULT_TIMEOUT_MS=900000
BASH_MAX_TIMEOUT_MS=1800000
```

The scraper also takes `--run-deadline-s`, which cancels roasters still in
flight when the budget runs out and keeps everything already collected. The
skill passes `--run-deadline-s 600`. Without it, one tarpitting host can
serialize for hours: per-host concurrency is 1, and the per-URL worst case is
three attempts at a 120-second deadline each.

No secrets are needed. Do not add a GitHub token; the session's GitHub identity
is provided by the platform.

## 3. Dependencies

Nothing to configure. `.claude/settings.json` registers a `SessionStart` hook
that runs `scripts/cloud_setup.sh`, which installs `httpx[http2]` only when
`CLAUDE_CODE_REMOTE=true` and only when the import is missing. It takes about a
second and never fails the session. A local checkout is untouched - you manage
your own venv.

If you would rather bake it into the environment snapshot, put
`python3 -m pip install -r requirements.txt` in the environment's **Setup
script** field instead; snapshotted setup is cached, a hook is not. You must do
that for a session with more than one repository: a multi-repo cloud session
loads neither hooks nor permission rules from any repo's `.claude/settings.json`.

## 4. A monthly routine

The scouting cadence is about once a month. Routines hold the schedule on your
claude.ai account, not in this repo.

Create one from the CLI with `/schedule`, or at claude.ai/code/routines. There is
no monthly preset, so pick the closest one and then set a cron expression with
`/schedule update` - for example `7 9 1 * *` for 09:07 on the 1st. An off-hour
minute avoids the on-the-hour queue.

The routine prompt must be self-contained, because nobody is watching:

```text
/coffee-scout

Monthly coffee scouting run. Scrape all configured roasters, then reply in the
session with the short Chinese scouting note the skill defines.

- Read purchase-journal.md first and check every watchlist item.
- Use --new-window-days 35 so the window matches the monthly cadence.
- If the scraper exits 3, or more than half the roasters return nothing, still
  reply: say what was collected, name the cause from logs/coffee_watch.log, and
  do not present an empty catalog as a quiet month.
- Do not write report files, do not commit, and do not push.
- Nobody is reading live: close with what to watch next instead of a question.
```

Pick the repository's default branch for the routine. A routine clones the
default branch unless the prompt names another one, so merge changes to `main`
before relying on them.

Each run shows up on the routine's page and in the session list; the reply is in
the run's transcript. A green run status means the session started and exited,
not that the scrape worked - which is why the reply is required to carry its own
evidence line.

## What a cloud run cannot do

- **No history.** `logs/seen_products.db` and previous `reports/*.json` are
  gitignored, so a cloud run has no baseline. `is_new` still works, because it
  comes from the roaster's own `published_at` (16 of 17 roasters are Shopify),
  but sold-out, restock and price-move deltas need two catalogs in one
  workspace, and `first_seen_at` is always "this run". The brief states this
  rather than guessing, and the skill is required to pass that on.
- **More bot-gating, probably.** Storefronts that already rate-limit the project
  User-Agent from a residential IP are no friendlier to a datacenter one. See
  [shopify-bot-gating.md](shopify-bot-gating.md). Expect fewer page-verified
  sizes in a cloud run than locally.
- **No `gh pr create`.** The GitHub proxy rejects GraphQL, so `gh pr`/`gh issue`
  fail; REST through `gh api repos/{owner}/{repo}/pulls` works, as does the
  session's own Create PR button. Not needed for scouting, which writes nothing.
