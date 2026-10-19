"""Guards for the cloud session contract in AGENTS.md.

A cloud session is a fresh clone in a disposable VM with no prior run state, so
the things that make scouting work have to be committed and have to be findable
at the paths Claude Code actually scans. Each assertion here failed, or would
have failed silently, at some point in this repo's history.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = PROJECT_ROOT / ".claude" / "skills" / "coffee-scout"
SKILL_MD = SKILL_DIR / "SKILL.md"
SETTINGS = PROJECT_ROOT / ".claude" / "settings.json"
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "cloud_setup.sh"

# Output the pivot removed. These filenames were a contract; a stray mention
# would pull the formal report structure back into the reply.
RETIRED_ARTIFACTS = (
    "z-digest.md",
    "z-roaster-digest.md",
    "z-new-digest.md",
    "z-purchase-report.md",
    "digest-prompts.md",
)


def _frontmatter(text: str) -> dict[str, str]:
    assert text.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
    _, block, _ = text.split("---\n", 2)
    fields: dict[str, str] = {}
    key = ""
    for line in block.splitlines():
        if line.startswith(" ") and key:
            fields[key] = f"{fields[key]} {line.strip()}".strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            fields[key] = value.strip()
    return fields


def test_skill_is_a_real_file_claude_code_scans():
    assert SKILL_MD.is_file(), "the skill must live at .claude/skills/coffee-scout/"
    assert not SKILL_DIR.is_symlink(), (
        "a symlinked skill dir needs core.symlinks on Windows clones; keep it a real "
        "directory"
    )
    assert not (PROJECT_ROOT / ".agents").exists(), (
        "the .agents/ indirection was retired with the portable-skill layout"
    )


def test_skill_frontmatter_drives_invocation():
    fields = _frontmatter(SKILL_MD.read_text(encoding="utf-8"))
    assert fields.get("name") == SKILL_DIR.name
    description = fields.get("description", "").lstrip(">").strip()
    assert description, "the description is what triggers implicit invocation"
    assert "scheduled" in description.lower(), (
        "a scheduled run has to recognize this skill as its own job"
    )


def test_claude_md_imports_agents_md_instead_of_forking_it():
    body = (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in body
    # Maintainer comments are stripped before Claude Code loads the file.
    without_comments = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    guidance = [
        line
        for line in without_comments.splitlines()
        if line.strip() and not line.strip().startswith("@")
    ]
    assert not guidance, "keep repository guidance in AGENTS.md"


def test_the_reply_is_the_only_deliverable():
    skill = SKILL_MD.read_text(encoding="utf-8")
    for artifact in RETIRED_ARTIFACTS:
        assert artifact not in skill, f"{artifact} is retired output"
    assert "Do not write report files" in skill


def test_skill_keeps_the_rules_that_prevent_wrong_recommendations():
    """Each phrase stands for a misread with real consequences."""
    skill = SKILL_MD.read_text(encoding="utf-8")
    for required in (
        "BULK-ONLY",  # wholesale-only bag read as a price rise
        "size-unverified",  # API weight shown as a confirmed size
        "size-from-api",  # API weight shown as a confirmed size
        "republish",  # a sale's timestamp bump read as a drop
        "page-blocked",  # access-gated page read as deleted
        "SOLD-OUT",  # unbuyable product recommended
        "Coverage is part of the answer",  # gated roaster read as a weak lineup
        "UNTRUSTED_SCRAPED_TEXT",  # prompt injection via roaster copy
        "purchase-journal.md",  # the only durable state
        "Flavour notes describe, they do not rank",  # copywriting is not quality
        "competition result or third-party placing",  # the hardest-to-fake signal
    ):
        assert required in skill, f"{required!r} must stay in the skill"


def test_skill_points_at_the_brief_not_the_raw_catalog():
    skill = SKILL_MD.read_text(encoding="utf-8")
    assert "coffee_watch.brief" in skill
    assert "must not be loaded" in skill, (
        "the skill has to say the combined catalog is too large to read directly"
    )


def test_committed_settings_cover_the_scout_commands():
    data = json.loads(SETTINGS.read_text(encoding="utf-8"))
    allow = data["permissions"]["allow"]
    # .claude/settings.local.json is not read in a cloud session, so anything the
    # skill runs has to be allowed from the committed file.
    for command in ("scrape_coffee.py", "coffee_watch.brief", "coffee_watch.hosts"):
        assert any(command in rule for rule in allow), f"{command} needs an allow rule"
    assert "deny" not in data.get("permissions", {}) or data["permissions"]["deny"]


def test_session_start_hook_installs_dependencies():
    data = json.loads(SETTINGS.read_text(encoding="utf-8"))
    hooks = data["hooks"]["SessionStart"]
    commands = [
        hook["command"] for group in hooks for hook in group["hooks"]
    ]
    assert any("cloud_setup.sh" in command for command in commands)
    assert all(hook.get("type") == "command" for group in hooks for hook in group["hooks"])

    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    assert SETUP_SCRIPT.stat().st_mode & 0o111, "the hook script must be executable"
    assert "CLAUDE_CODE_REMOTE" in script, "a local checkout must be left alone"
    assert "requirements.txt" in script, (
        "the http2 extra is mandatory; a bare 'pip install httpx' breaks the client"
    )
    assert script.rstrip().endswith("exit 0"), (
        "a failing SessionStart hook must not be able to break the session"
    )


def test_requirements_stay_installable_in_seconds():
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "httpx[http2]" in text, "runner.py creates the client with http2=True"
    specs = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(specs) == 1, (
        "keep the cloud install to one pure-Python dependency; anything needing a "
        "system package or compilation breaks the SessionStart hook budget"
    )


def test_cloud_setup_doc_covers_what_the_repo_cannot_ship():
    doc = (PROJECT_ROOT / "docs" / "cloud-setup.md").read_text(encoding="utf-8")
    for topic in (
        "coffee_watch.hosts",  # the network allowlist
        "BASH_DEFAULT_TIMEOUT_MS",  # the scrape outruns the default timeout
        "/coffee-scout",  # the routine prompt
        "host_not_allowed",  # the symptom of a missing allowlist
    ):
        assert topic in doc, f"{topic} is environment setup the repo cannot do itself"
