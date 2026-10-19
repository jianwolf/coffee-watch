#!/usr/bin/env bash
# Install the scraper's dependency in a cloud session.
#
# A cloud session starts from a fresh clone with no .venv, so httpx is missing
# and the first scrape dies on ModuleNotFoundError. This runs as a SessionStart
# hook (see .claude/settings.json) and does nothing in a local checkout, where
# the developer owns their own environment.
#
# It must never fail the session: every path exits 0.
set -u

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    exit 0
fi

# Hooks run in the session's current directory; anchor to the repo so the
# relative requirements.txt below resolves wherever the hook fires from.
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

PY="${PYTHON:-python3}"

# Silence on success is deliberate: a SessionStart hook's stdout is added to
# the session context, so only an actionable failure is worth saying.
if "$PY" -c 'import httpx, h2' >/dev/null 2>&1; then
    exit 0
fi

# requirements.txt, not a bare "pip install httpx": the http2 extra is
# mandatory because the client is created with http2=True.
#
# --break-system-packages is for Debian/Ubuntu's PEP 668 marker, which makes a
# plain install refuse on the system interpreter. The cloud VM is disposable and
# single-purpose, so installing into it is the intended behaviour there.
if ! "$PY" -m pip install -q -r requirements.txt >/dev/null 2>&1; then
    "$PY" -m pip install -q --break-system-packages -r requirements.txt \
        >/dev/null 2>&1 || true
fi

if ! "$PY" -c 'import httpx, h2' >/dev/null 2>&1; then
    echo "coffee-watch: could not install httpx[http2]; run" \
        "'python3 -m pip install -r requirements.txt' before scraping"
fi
exit 0
