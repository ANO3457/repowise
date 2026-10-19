"""Consent for repointing the one global ``repowise`` MCP entry.

Claude Code (``~/.claude/settings.json``) and Claude Desktop each hold a
single ``repowise`` server key, so registering from a new repo moves every
agent session on the machine to it. :func:`read_global_mcp_target` is the
read-only probe a start screen can show; :func:`confirm_global_mcp` is the one
place init asks.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from repowise.cli.editor_integrations import claude_config

NOT_SET = "not set"
#: Re-runs the user-level writes a declined init skipped.
REGISTER_LATER_COMMAND = "repowise agents add --target=claude-code --scope=user"


@dataclass(frozen=True)
class GlobalMcpEntry:
    """One client's stored ``repowise`` entry."""

    client: str
    config_path: Path
    target: str | None
    command: str | None


@dataclass(frozen=True)
class GlobalMcpTarget:
    """Where the global entry points now, and where registration would move it."""

    new_target: str
    new_command: str
    entries: tuple[GlobalMcpEntry, ...]

    @property
    def repoints(self) -> bool:
        """Whether any client would move to a different repo (or gain an entry)."""
        return any(not _same_path(e.target, self.new_target) for e in self.entries)

    @property
    def command_changes(self) -> bool:
        return any(e.command != self.new_command for e in self.entries if e.target)

    def summary_lines(self) -> list[str]:
        """``global entry now points at X, will point here`` for each distinct X."""
        currents = {e.target or NOT_SET for e in self.entries}
        if len(currents) <= 1:
            clients = " and ".join(e.client for e in self.entries) or "Claude Code"
            now = next(iter(currents), NOT_SET)
            return [
                f"Global repowise MCP entry ({clients}) now points at {now}, "
                f"will point here: {self.new_target}"
            ]
        lines = [f"Global repowise MCP entry will point here: {self.new_target}"]
        lines += [f"    {e.client} now points at {e.target or NOT_SET}" for e in self.entries]
        return lines


def _same_path(a: str | None, b: str) -> bool:
    if a is None:
        return False
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def read_global_mcp_target(repo_path: Path) -> GlobalMcpTarget:
    """Read-only: what each global config holds and what init would write.

    Lists the clients registration would write: Claude Code always, Claude
    Desktop only when installed, the same rule the writers apply.
    """
    new_entry = claude_config.generate_mcp_config(
        claude_config._resolve_mcp_target(repo_path),
        command=claude_config.resolve_repowise_command(),
    )["mcpServers"]["repowise"]

    configs: list[tuple[str, Path]] = [
        ("Claude Code", claude_config._claude_code_settings_path())
    ]
    desktop = claude_config._claude_desktop_config_path()
    if desktop is not None and desktop.parent.exists():
        configs.append(("Claude Desktop", desktop))

    entries = []
    for client, path in configs:
        stored = claude_config._stored_repowise_entry(path) or {}
        entries.append(
            GlobalMcpEntry(
                client=client,
                config_path=path,
                target=claude_config._entry_repo_target(stored),
                command=stored.get("command"),
            )
        )
    return GlobalMcpTarget(
        new_target=claude_config._entry_repo_target(new_entry) or "",
        new_command=str(new_entry.get("command")),
        entries=tuple(entries),
    )


def confirm_global_mcp(
    console_obj: Any,
    repo_path: Path,
    *,
    yes: bool = False,
    no_editor_setup: bool = False,
) -> bool:
    """Whether init may write user-level client config for *repo_path*.

    Asks (default yes) only when the entry would move and someone can answer.
    ``--yes`` and a non-terminal stdin keep registering, the documented
    scripted path, but still print the old and new target. A decline leaves
    every user-level file alone; project files are written either way.
    """
    from repowise.cli.editor_setup import is_editor_setup_disabled

    if is_editor_setup_disabled(no_editor_setup):
        return False
    try:
        target = read_global_mcp_target(repo_path)
    except Exception:
        # A probe failure must not block registration the user did not refuse.
        return True

    if not target.repoints:
        if target.command_changes:
            console_obj.print(
                f"  [dim]Global repowise MCP entry stays on {target.new_target}; "
                f"its command moves to {target.new_command}.[/dim]"
            )
        return True

    for line in target.summary_lines():
        console_obj.print(f"  {line}")
    if yes or not sys.stdin.isatty():
        return True
    try:
        agreed = click.confirm("  Point it here?", default=True)
    except (click.Abort, EOFError):
        agreed = False
    if not agreed:
        console_obj.print(
            "  [dim]Global Claude config left as is. Project files still connect "
            f"agents opened in this repo. Run '{REGISTER_LATER_COMMAND}' to point "
            "it here later.[/dim]"
        )
    return agreed
