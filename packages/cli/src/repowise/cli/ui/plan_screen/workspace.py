"""The workspace plan screen: the repos found, which ones run, and the shared plan."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from repowise.cli.ui.brand import BRAND_STYLE, OK, WARN
from repowise.cli.ui.plan_screen.render import render_plan_body
from repowise.cli.ui.plan_screen.state import (
    PlanState,
    RunShape,
    eta_phrase,
    eta_seconds,
    model_calls_for,
)
from repowise.cli.ui.repo_scanner import RepoScanInfo
from repowise.cli.ui.workspace_selection import (
    display_path,
    interactive_primary_select,
    parse_selection,
)

#: Rows the screen lists before it summarises the rest; ``r`` lists them all.
ROWS_SHOWN = 12

#: What ``e`` opens on a workspace: one set of answers for every repo.
WORKSPACE_EDIT_PREVIEW = (
    "scope, excludes, commit limit, embedder and sync for every repo, plus "
    "concurrency, style and language with a model"
)

REPOS_KEY = ("r", "choose repos")


@dataclass(frozen=True)
class WorkspaceRepo:
    """One repo the workspace scan found, with its pre-scan."""

    repo: Any
    scan: RepoScanInfo


@dataclass
class WorkspacePlanState:
    """One render of the workspace screen: the repos, the pick and the plan."""

    root: Path
    repos: list[WorkspaceRepo]
    #: Zero-based indices into *repos*, in workspace order.
    selected: list[int]
    primary: str
    plan: PlanState
    #: Sizes each repo's model calls, as the combined price does.
    onboarding: bool = True

    @property
    def picked(self) -> list[WorkspaceRepo]:
        return [self.repos[i] for i in self.selected]

    def shape(self, onboarding: bool = True, concurrency: int = 10) -> RunShape:
        """The selected repos, priced together and each planned as its own repo."""
        scans = [r.scan for r in self.picked] or [RepoScanInfo()]
        return RunShape(scans[0], onboarding, concurrency, also=tuple(scans[1:]))


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _eta_line(state: WorkspacePlanState) -> str:
    low = high = 0.0
    for repo in state.picked:
        lo, hi = eta_seconds(replace(state.plan, scan=repo.scan, chosen=None))
        low, high = low + lo, high + hi
    what = "to index and build the documentation"
    chosen = state.plan.chosen
    if chosen is not None:
        low, high = low + chosen.minutes * 60, high + chosen.minutes * 60
        what = "to index and write the documentation"
    phrase = eta_phrase(low, high)
    return f"{phrase[0].upper()}{phrase[1:]} {what} for {_plural(len(state.picked), 'repo')}."


def _shows_calls(state: WorkspacePlanState) -> bool:
    plan = state.plan
    return plan.run_mode != "fast" and (plan.chosen is not None or plan.offer.price is not None)


def repo_table(state: WorkspacePlanState, limit: int | None = ROWS_SHOWN) -> Table:
    """The repo list: pick mark, number, name, path, files, status."""
    table = Table(box=None, padding=(0, 1), pad_edge=False, show_header=True, header_style="dim")
    table.add_column("", no_wrap=True)
    table.add_column("#", style=BRAND_STYLE, justify="right", no_wrap=True)
    table.add_column("Repository", style="bold", max_width=24, overflow="ellipsis", no_wrap=True)
    table.add_column("Path", style="dim", max_width=30, overflow="ellipsis", no_wrap=True)
    table.add_column("Files", justify="right", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    calls = _shows_calls(state)
    if calls:
        table.add_column("Model calls", justify="right", no_wrap=True)
    picked = set(state.selected)
    rows = state.repos if limit is None else state.repos[:limit]
    for idx, row in enumerate(rows):
        repo = row.repo
        status = Text("indexed", style=OK) if repo.has_repowise else Text("new", style="dim")
        if repo.is_submodule:
            status.append(" (submodule)", style="dim")
        if idx in picked and repo.alias == state.primary:
            status.append(" · primary", style="bold")
        cells: list[Any] = [
            Text("[x]" if idx in picked else "[ ]"),
            str(idx + 1),
            Text(repo.name),
            Text(display_path(repo, state.root)),
            f"{row.scan.total_files:,}",
            status,
        ]
        if calls:
            n = model_calls_for(RunShape(row.scan, state.onboarding)) if idx in picked else 0
            cells.append(f"about {n:,}" if idx in picked else "")
        table.add_row(*cells)
    return table


def _hidden_line(state: WorkspacePlanState) -> str | None:
    hidden = len(state.repos) - ROWS_SHOWN
    if hidden <= 0:
        return None
    return (
        f"[dim]{_plural(hidden, 'more repo')} not shown; "
        f"{len(state.selected):,} of {len(state.repos):,} selected in all. "
        "r lists every repo.[/dim]"
    )


def render_workspace_plan(console: Console, state: WorkspacePlanState) -> None:
    """Print the workspace plan screen for *state*."""
    files = sum(r.scan.total_files for r in state.picked)
    console.print(
        f"  [bold]{escape(state.root.name or str(state.root))}[/bold] · "
        f"{_plural(len(state.repos), 'repo')} found · {len(state.selected):,} selected "
        f"({_plural(files, 'file')})",
        highlight=False,
    )
    console.print(f"  [dim]{escape(str(state.root))}[/dim]", highlight=False)
    console.print(f"  {_eta_line(state)}", highlight=False)
    console.print()
    console.print(Padding(repo_table(state), (0, 0, 0, 2)), highlight=False)
    hidden = _hidden_line(state)
    if hidden:
        console.print(f"  {hidden}", highlight=False)
    render_plan_body(
        console,
        state.plan,
        extra_keys=(REPOS_KEY,),
        edit_preview=WORKSPACE_EDIT_PREVIEW,
    )


def choose_repos(
    console: Console, state: WorkspacePlanState, ask: Callable[[], str] | None = None
) -> None:
    """``r``: every repo listed, then which to index and which is the primary.

    Enter keeps the current pick. EOFError propagates, as on the key read.
    """
    if ask is None:

        def ask() -> str:
            return Prompt.ask("  Repos to index", default="", show_default=False, console=console)

    console.print()
    console.print(Padding(repo_table(state, limit=None), (0, 0, 0, 2)), highlight=False)
    console.print(
        "  [dim]Numbers (1,3), ranges (2-5) or all. Enter keeps the current choice.[/dim]"
    )
    count = len(state.repos)
    while True:
        raw = ask().strip().lower()
        if not raw:
            break
        picked = list(range(count)) if raw == "all" else parse_selection(raw, count)
        if picked:
            state.selected = sorted(picked)
            break
        if raw == "none":
            console.print(f"  [{WARN}]Choose at least one repo, or press q on the plan to quit.[/]")
        else:
            console.print(f"  [{WARN}]Invalid selection. Use numbers (1,3), ranges (2-5) or all.[/]")
    repos = [r.repo for r in state.picked]
    state.primary = (
        interactive_primary_select(console, repos, current=state.primary)
        if len(repos) > 1
        else repos[0].alias
    )
