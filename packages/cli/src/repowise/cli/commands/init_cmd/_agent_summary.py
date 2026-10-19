"""How ``repowise init`` writes for a reader that is not a person.

Off a terminal, lines stay within :data:`~repowise.cli.output.LINE_OUTPUT_WIDTH`.
Under ``--progress json``, stdout is the :mod:`repowise.cli.agent_events`
stream, and every exit ends it with a ``summary`` line.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import click

from repowise.cli import agent_events
from repowise.cli.helpers import console, load_config, load_state
from repowise.cli.output import line_output_width

from ._ingest import RESUME_COMMAND
from ._options import InitRun

SERVE_COMMAND = "repowise serve"


def mode_of(scope: Mapping[str, Any]) -> dict[str, Any]:
    """``{"key", "label", "run_mode"}`` for a resolved index scope.

    ``key`` is ``index_scope.content_provenance``: ``model``, ``template``,
    ``none`` (fast, no documentation), ``mixed`` or ``unknown``.
    """
    key = scope.get("content_provenance") or "unknown"
    if key == "unknown" and scope.get("run_mode") == "fast":
        key = "none"
    provider = scope.get("provider") if isinstance(scope.get("provider"), Mapping) else {}
    model = provider.get("model")
    # Ceiling: the CLI's naming table is not on this branch yet; read these
    # labels from it once it is, so the two cannot drift.
    if key in ("model", "mixed"):
        label = f"Written by {model}" if model else "Written by a model"
    else:
        label = {"template": "Built from code", "none": "Index only"}.get(key, "Mode unknown")
    return {"key": key, "label": label, "run_mode": scope.get("run_mode")}


def _stored_mode(repo_path: Path) -> dict[str, Any] | None:
    """The mode state.json records, or None when this repo has no index yet."""
    from repowise.core.index_scope import resolve_index_scope

    # Runs inside the exit handlers: a corrupt state or config must not cost the summary.
    try:
        state = load_state(repo_path)
        if not state.get("index_scope"):
            return None
        return mode_of(resolve_index_scope(state, load_config(repo_path)))
    except Exception:
        return None


def _relative(repo_path: Path, paths: list[Path]) -> list[str]:
    out = []
    for path in paths:
        try:
            out.append(path.relative_to(repo_path).as_posix())
        except ValueError:
            out.append(str(path))
    return out


def run_summary(run: InitRun, gaps: list[str]) -> dict[str, Any]:
    """The summary of a run that finished: ready, or degraded by *gaps*."""
    result = run.result
    pages = getattr(result, "generated_pages", None) or []
    cost = getattr(result, "llm_cost_usd", None)
    setup = run.setup_outcome
    restart = bool(
        setup is not None
        and not setup.editor_setup_disabled
        and setup.claude_code_connected
        and (setup.first_index or run.mcp_repointed)
    )
    next_commands = [command for command, _gloss in run.next_steps]
    if SERVE_COMMAND not in next_commands:
        next_commands.insert(0, SERVE_COMMAND)
    return agent_events.summary_fields(
        agent_events.STATUS_DEGRADED if gaps else agent_events.STATUS_READY,
        mode=_stored_mode(run.repo_path),
        restart_required=restart,
        spend={
            "tokens": sum(getattr(p, "total_tokens", 0) or 0 for p in pages),
            "usd": round(cost or 0.0, 4),
        },
        warnings=[*run.run_warnings, *gaps],
        files_written=_relative(run.repo_path, run.files_written),
        next_commands=next_commands,
    )


def _exit_summary(status: str, repo_path: Path | None) -> dict[str, Any]:
    """The summary when the run ended before :func:`run_summary` could be built."""
    if status == agent_events.STATUS_INTERRUPTED:
        return agent_events.summary_fields(status, next_commands=[RESUME_COMMAND])
    mode = _stored_mode(repo_path) if repo_path is not None else None
    next_commands = [SERVE_COMMAND] if mode is not None and status != agent_events.STATUS_FAILED else []
    return agent_events.summary_fields(status, mode=mode, next_commands=next_commands)


@contextlib.contextmanager
def init_output(progress: str, repo_path: Path | None) -> Iterator[None]:
    """Cap line width off a terminal; under ``--progress json``, own stdout until exit."""
    if progress != "json":
        with line_output_width(console, sys.stdout):
            yield
        return
    with agent_events.open_stream(console) as events:
        try:
            yield
        except SystemExit as exc:
            events.summary(_exit_summary(agent_events.status_for_exit(exc.code), repo_path))
            raise
        except click.exceptions.Exit as exc:
            events.summary(_exit_summary(agent_events.status_for_exit(exc.exit_code), repo_path))
            raise
        except click.ClickException as exc:
            events.summary(
                agent_events.summary_fields(agent_events.STATUS_FAILED, error=exc.format_message())
            )
            raise
        except BaseException as exc:
            events.summary(
                agent_events.summary_fields(
                    agent_events.STATUS_FAILED, error=str(exc) or type(exc).__name__
                )
            )
            raise
        events.summary(_exit_summary(agent_events.STATUS_READY, repo_path))
