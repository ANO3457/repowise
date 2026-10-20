"""Which repos a multi-repo ``repowise init`` indexes, and which one is primary."""

from __future__ import annotations

import sys
from typing import Any

import click

from repowise.cli.helpers import console
from repowise.cli.ui.brand import WARN
from repowise.cli.ui.workspace_selection import (
    interactive_primary_select,
    interactive_repo_select,
    match_repo_names,
)

#: Names shown in the all-repos line before it says "and N more".
_NAMES_SHOWN = 10


def split_repo_names(raw: str | None) -> list[str]:
    """``--repos a,b , c`` as ``["a", "b", "c"]``; empty when the flag is absent."""
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def select_workspace_repos(
    scan: Any, *, repo_names: list[str], select_all: bool
) -> tuple[list[Any], str, Any] | None:
    """The repos to index, the primary alias and repo; None when none were selected.

    ``--repos`` wins over everything and never prompts. Otherwise ``--yes``,
    ``--all`` or a non-terminal stdin take every detected repo (the first is
    primary): asking a pipe would end the run with an EOFError before anything
    was done.
    """
    root = scan.root
    if repo_names:
        try:
            selected = match_repo_names(scan.repos, repo_names, root)
        except ValueError as exc:
            raise click.UsageError(f"--repos: {exc}") from exc
        console.print(f"  Primary: [bold]{selected[0].alias}[/bold] (first in --repos)\n")
        return selected, selected[0].alias, selected[0]

    select_all = select_all or not sys.stdin.isatty()
    if select_all:
        selected = list(scan.repos)
        console.print(all_repos_line(selected))
    else:
        selected = interactive_repo_select(console, scan.repos, root)
    if not selected:
        console.print(f"[{WARN}]No repositories selected. Aborting.[/]")
        return None
    primary_alias = (
        selected[0].alias if select_all else interactive_primary_select(console, selected)
    )
    primary_repo = next((r for r in selected if r.alias == primary_alias), selected[0])
    return selected, primary_alias, primary_repo


def all_repos_line(repos: list[Any]) -> str:
    """One line naming every repo a promptless run takes, and how to pick fewer."""
    names = [r.alias for r in repos]
    shown = ", ".join(names[:_NAMES_SHOWN])
    if len(names) > _NAMES_SHOWN:
        shown += f" and {len(names) - _NAMES_SHOWN} more"
    example = ",".join(names[:2])
    return (
        f"  Indexing all {len(names)} repos: {shown}.\n"
        f"  [dim]Pick some with --repos {example}[/dim]\n"
    )
