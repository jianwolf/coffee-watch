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

Create (or edit) an environment at claude.ai/code and set network access wide
enough to reach the roasters. **Full** is what has been verified to work; see
the measured results below before choosing **Custom**.

For Custom, keep "include default list of common package managers" checked and
paste the host list:

```bash
python -m coffee_watch.hosts
```

That prints every host the scrape needs: the configured `base_url` hosts, the
hosts the last catalog's product URLs actually resolved to (one roaster is
configured on a `myshopify.com` entry point that redirects elsewhere), and an
apex wildcard where the host is the roaster's own domain. Re-run it after
changing `config/roasters.json` and update the environment.

### What a blocked run looks like

Measured on 2026-10-19, same repo, same routine, twenty minutes apart:

| Network access | Result |
| :--- | :--- |
| Custom, with the generated host list | 0 of 17 roasters; every `robots.txt` request refused; exit 3 |
| Full | 9 of 17 roasters, 173 products, 112 s; the other 8 were roaster `429`s |

Two things about the blocked run are worth knowing, because both mislead:

- **There was no `x-deny-reason` header.** The documented form is a 403 response
  carrying it. What actually happened is that the proxy refused the CONNECT, so
  there was no response at all -- httpx raises `ProxyError("403 Forbidden")`.
  `coffee_watch/network.py` now logs that case as `Egress proxy refused`.
- **It fails at `robots.txt`, before any catalog request.** RFC 9309 makes an
  unreachable `robots.txt` a complete disallow, so one refused connection per
  host silently removes that roaster from the run. A fully blocked run therefore
  looks like 17 polite skips rather than 17 errors.

Whether Custom failed because the list was incomplete or because of how the
environment applied it is not established here; only that Full worked and Custom
did not. Custom remains the better shape if you can get it working -- it keeps
the blast radius of a scraper bug to coffee shops -- so re-check it with a
`Run now` before trusting it, and read the run's own coverage line rather than
the green status dot.

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

## 4. Getting the note to your phone

A routine cannot notify you on its own. Push notifications belong to Remote
Control, which runs `claude` on your own machine; a routine is a cloud session,
so its `PushNotification` call is accepted and delivered nowhere. Verified
twice: both runs logged `Mobile push requested.` and nothing arrived. There is
no documented completion notification for a scheduled cloud run at all.

Outbound HTTP is the one channel a cloud run does have, so delivery is
something the run sends itself. `scripts/notify_discord.py` posts a note to a
Discord channel webhook - stdlib only, split to Discord's 2000-character limit,
which fits a whole scouting note rather than a 200-character headline.

Create a webhook in Discord (channel settings -> Integrations -> Webhooks ->
New Webhook), set that channel's notifications to All Messages, and give the
routine the URL. **The URL is a credential and must never be committed** -
anyone holding it can post to that channel, and this repository is public.
`tests/test_notify_discord.py` fails if a real webhook URL appears anywhere in
the tree. Put it in the routine prompt, or as a `COFFEE_DISCORD_WEBHOOK`
environment variable on the cloud environment, remembering that anyone who can
use that environment can read its variables.

Link previews are suppressed by default (`flags: 4`). Discord otherwise
unfurls every bare URL into a card carrying the roaster's own marketing copy
and a product photo, which buries a fifteen-line note under three screens of
sales text - the copy this project deliberately ranks below hard evidence.
`--embeds` opts back in. Write links as `[name](url)` so a URL costs part of a
line rather than a line of its own.

Check the wiring without sending anything:

```bash
echo "hello" | python3 scripts/notify_discord.py --dry-run
```

Exit codes are distinct on purpose: `3` means no webhook was configured, `1`
means the post failed. A run that scouted correctly but could not deliver is
not the same failure as a run that found nothing.

## 4b. The routine

The routine runs daily at 13:07 UTC. Daily is justified by one thing and it is
worth stating: scarcity. SEY's 2026 Gales Gesha was listed on 21 October and
sold out inside two days, so a weekly cadence would have missed it entirely.
Routines hold the schedule on your claude.ai account, not in this repo.

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
evidence line. Both runs on 2026-10-19 reported `success`; one of them had
collected nothing.

Grant the routine `Bash`, `Read`, `Glob`, `Grep` and `Skill`, and withhold
`Write` and `Edit`: the scraper writes its JSON through `python`, so the only
thing those two would add is the ability to leave files in the repo, which a
scheduled run must not do.

## What a cloud run cannot do

- **No history.** `logs/seen_products.db` and previous `reports/*.json` are
  gitignored, so a cloud run has no baseline. `is_new` still works, because it
  comes from the roaster's own `published_at` (16 of 17 roasters are Shopify),
  but sold-out, restock and price-move deltas need two catalogs in one
  workspace, and `first_seen_at` is always "this run". The brief states this
  rather than guessing, and the skill is required to pass that on.
- **The same bot-gating as anywhere else.** Measured 2026-10-19: a cloud run and
  a residential one started 90 seconds apart returned 3/17 and 4/17 roasters,
  and both degraded together despite very different request histories. Neither
  the egress IP nor your own request volume is the variable. See
  [shopify-bot-gating.md](shopify-bot-gating.md).
- **No `gh pr create`.** The GitHub proxy rejects GraphQL, so `gh pr`/`gh issue`
  fail; REST through `gh api repos/{owner}/{repo}/pulls` works, as does the
  session's own Create PR button. Not needed for scouting, which writes nothing.

## 5. Asking questions back, from the Discord channel

The webhook in section 4 only posts. Reading the channel needs a **bot
application**, and `scripts/discord_listen.py` turns what you type there into
a headless Claude Code turn in this repo, then posts the answer back. It runs
wherever you keep a machine awake - a Mac mini, not the cloud, because a
routine run is a disposable VM that exists for three minutes.

It polls `GET /channels/{id}/messages?after=` every few seconds rather than
holding a gateway socket, which keeps it stdlib-only like the notifier.

### Why it only answers one person

Whatever is typed in the channel becomes the prompt for a Claude Code session
on your machine, running with `--permission-mode bypassPermissions`. The
author check is the entire boundary between "I can drive my Mac mini from my
phone" and "anyone holding the invite link can". The script refuses to start
without `DISCORD_OWNER_ID` for that reason, and it ignores every other author
and every bot - including the scouting webhook, which would otherwise answer
itself in a loop.

### Setup

1. **Create the bot.** discord.com/developers → New Application → Bot. Copy
   the token once; it is shown only at creation.
2. **Enable MESSAGE CONTENT.** Bot → Privileged Gateway Intents → Message
   Content Intent. This gates message text across Discord's REST API as well
   as the gateway: without it, `content` arrives as an empty string and the
   listener sees every message as blank. For an app in under 100 servers you
   can toggle it yourself.
3. **Invite it.** OAuth2 → URL Generator → scopes `bot`, permissions
   `View Channels`, `Send Messages`, `Read Message History`. Open the URL and
   add it to your server.
4. **Get the two ids.** Discord Settings → Advanced → Developer Mode, then
   right-click the channel → Copy Channel ID, and right-click yourself →
   Copy User ID.
5. **Check it once in the foreground**, which drains the channel a single
   time and exits:

```bash
DISCORD_BOT_TOKEN='...' DISCORD_CHANNEL_ID='...' DISCORD_OWNER_ID='...' \
    python3 scripts/discord_listen.py --once
```

6. **Keep it running** with launchd. Put the three values in
   `~/Library/LaunchAgents/coffee-watch-discord.plist` as
   `EnvironmentVariables`, point `ProgramArguments` at
   `/usr/bin/python3 <repo>/scripts/discord_listen.py`, set `RunAtLoad` and
   `KeepAlive` to true, and `launchctl load` it. Never commit that plist: it
   holds the token.

### The authentication it runs on

A subscription OAuth session, not an API key - so questions come out of the
plan rather than metered billing. The cost is that the session expires if the
machine goes long enough without using `claude`, and a 24/7 listener that
quietly stops answering is worse than one that says why. When a turn comes
back with an authentication error the listener posts that into the channel
instead of going silent; sign in again with `claude` on that machine.

### What it cannot do

It answers in the channel, with whatever `coffee-scout` and the repo can see.
It is not the scheduled routine and does not replace it: the 8:07 note still
comes from the cloud routine, which needs no machine of yours to be awake.
