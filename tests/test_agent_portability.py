"""Guards that keep the coffee-scout skill runnable in any agent.

The skill body is single-source at `.agents/skills/coffee-scout/SKILL.md`. Codex and
Cursor read that path directly; Claude Code reads it through `.claude/skills/`. These
tests fail when the layout drifts back toward one vendor.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = PROJECT_ROOT / ".agents" / "skills" / "coffee-scout"
SKILL_MD = SKILL_DIR / "SKILL.md"
CLAUDE_SKILL_DIR = PROJECT_ROOT / ".claude" / "skills" / "coffee-scout"

# Every doc that names one supported agent must name all of them, so no reader is
# nudged toward a default vendor.
AGENT_ALIASES = {
    "Codex": ("codex",),
    "Claude Code": ("claude code", "claude-code"),
    "Cursor": ("cursor",),
}

DOCS_MENTIONING_AGENTS = (
    "AGENTS.md",
    "README.md",
    ".agents/skills/coffee-scout/SKILL.md",
    ".agents/skills/coffee-scout/references/digest-prompts.md",
)

# Artifact names are part of the skill contract: a session in one agent must be able to
# read what another agent wrote.
REPORT_ARTIFACTS = (
    "reports/YYYYMMDD-z-digest.md",
    "reports/YYYYMMDD-z-roaster-digest.md",
    "reports/YYYYMMDD-z-new-digest.md",
    "reports/YYYYMMDD-z-purchase-report.md",
)

VENDOR_TOKENS = ("codex", "claude", "cursor", "openai", "anthropic", "grok", "gpt")


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


def test_skill_body_lives_at_the_cross_vendor_path():
    assert SKILL_MD.is_file(), "coffee-scout SKILL.md must live in .agents/skills/"
    assert not (PROJECT_ROOT / "skills").exists(), (
        "the legacy repo-root skills/ directory is not scanned by any agent"
    )


def test_skill_frontmatter_matches_the_portable_contract():
    fields = _frontmatter(SKILL_MD.read_text(encoding="utf-8"))
    assert fields.get("name") == SKILL_DIR.name, "frontmatter name must match the directory"
    description = fields.get("description", "").lstrip(">").strip()
    assert description, "frontmatter description drives implicit invocation in every agent"


def test_claude_code_entry_point_resolves_to_the_same_skill():
    resolved = CLAUDE_SKILL_DIR / "SKILL.md"
    assert resolved.is_file(), ".claude/skills/coffee-scout must expose the skill"
    assert resolved.read_text(encoding="utf-8") == SKILL_MD.read_text(encoding="utf-8"), (
        "the Claude Code entry point must not be a diverging copy of the skill"
    )


def test_claude_md_imports_agents_md_instead_of_forking_it():
    claude_md = PROJECT_ROOT / "CLAUDE.md"
    assert claude_md.is_file(), "Claude Code reads CLAUDE.md, not AGENTS.md"
    body = claude_md.read_text(encoding="utf-8")
    assert "@AGENTS.md" in body, "CLAUDE.md must import AGENTS.md"
    # Maintainer comments are stripped before Claude Code loads the file.
    without_comments = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    guidance = [
        line
        for line in without_comments.splitlines()
        if line.strip() and not line.strip().startswith("@")
    ]
    assert not guidance, "keep repository guidance in AGENTS.md, not duplicated in CLAUDE.md"


def test_skill_carries_no_vendor_specific_manifest():
    manifests = [
        path
        for path in SKILL_DIR.rglob("*")
        if path.is_file() and path.suffix in {".yaml", ".yml", ".json", ".toml"}
    ]
    assert not manifests, (
        f"SKILL.md frontmatter is the portable contract; drop vendor manifests: {manifests}"
    )


def test_report_artifact_names_are_vendor_neutral_and_declared():
    skill_text = SKILL_MD.read_text(encoding="utf-8")
    for artifact in REPORT_ARTIFACTS:
        assert artifact in skill_text, f"{artifact} must stay in the skill contract"
        assert not any(token in artifact for token in VENDOR_TOKENS), (
            f"{artifact} names a vendor"
        )


def test_docs_naming_one_agent_name_all_of_them():
    for rel in DOCS_MENTIONING_AGENTS:
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8").lower()
        mentioned = {
            name for name, aliases in AGENT_ALIASES.items() if any(a in text for a in aliases)
        }
        assert mentioned in (set(), set(AGENT_ALIASES)), (
            f"{rel} mentions {sorted(mentioned)} but not every supported agent"
        )
