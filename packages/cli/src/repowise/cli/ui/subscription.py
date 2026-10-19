"""Own-subscription providers: a logged-in Claude Code, Codex or OpenCode CLI.

Detected and offered, never picked on the user's behalf: the picker shows them
with a time and quota estimate, and a run that writes no documentation names
the one it found. Detection lives here so the picker, that note and the init
plan screen all ask the same question.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from repowise.core.agents.identity import identity_for_provider
from repowise.core.providers.llm.specs import PROVIDER_SPECS

#: Agent-CLI providers, in picker order.
AGENT_PROVIDERS: tuple[str, ...] = tuple(
    spec.name
    for spec in sorted(PROVIDER_SPECS.values(), key=lambda s: s.picker_rank or 0)
    if spec.agent and spec.picker_rank is not None
)

# One page is one CLI process and a full model turn. A serialized 68-page run
# took about an hour (docs/agent/CLAUDE_CODE_PROVIDER.md), so a minute a call
# is the planning figure. Rough on purpose: it sizes the wait, not a deadline.
SECONDS_PER_CALL = 60


@dataclass(frozen=True)
class SubscriptionContext:
    """What the picker knows about the run that opened it.

    *prefer*: agent CLIs set up in this repo, the first ready one becomes the
    default (see ``agent_providers_set_up``). *model_calls* and *concurrency*
    size the quota and time estimate; no estimate is shown without calls.
    *allow_skip*: a provider that is not set up offers to go on without a
    model, and the picker returns ``None`` for that answer.
    """

    prefer: tuple[str, ...] = ()
    model_calls: int | None = None
    concurrency: int = 10
    allow_skip: bool = False


def agent_cli_status(name: str) -> tuple[bool, bool]:
    """``(installed, ready)`` for the agent CLI behind provider *name*.

    Ready means signed in where the CLI can say so cheaply (``claude auth
    status``, ``codex login status``), else installed: a CLI without such a
    query surfaces a missing login on first use.
    """
    agent = identity_for_provider(name)
    if agent is None or not agent.is_installed():
        return False, False
    return True, agent.is_logged_in()


def ready_agent_providers() -> tuple[str, ...]:
    """Agent-CLI providers whose CLI is installed and signed in, in picker order.

    Each login check is a short subprocess, so they run side by side.
    """
    with ThreadPoolExecutor(max_workers=len(AGENT_PROVIDERS) or 1) as pool:
        ready = list(pool.map(lambda name: agent_cli_status(name)[1], AGENT_PROVIDERS))
    return tuple(name for name, ok in zip(AGENT_PROVIDERS, ready, strict=True) if ok)


def process_limit(provider_name: str) -> int:
    """How many CLI processes *provider_name* runs at once."""
    spec = PROVIDER_SPECS[provider_name]
    cls = getattr(importlib.import_module(spec.module_path), spec.class_name)
    return int(cls.process_limit())


def model_call_count(plans: Iterable[object]) -> int:
    """Model calls a generation plan makes: one per page a model writes."""
    from repowise.core.cost_estimator import STRUCTURAL_PAGE_TYPES

    return sum(
        int(getattr(plan, "count", 0))
        for plan in plans
        if getattr(plan, "page_type", None) not in STRUCTURAL_PAGE_TYPES
    )


def _parallel(provider_name: str, concurrency: int | None) -> int:
    limit = process_limit(provider_name)
    return max(1, min(concurrency, limit) if concurrency else limit)


def subscription_minutes(
    provider_name: str, model_calls: int, concurrency: int | None = None
) -> int:
    """Minutes a subscription run of *model_calls* takes, at least one."""
    parallel = _parallel(provider_name, concurrency)
    return max(1, math.ceil(math.ceil(model_calls / parallel) * SECONDS_PER_CALL / 60))


def subscription_quota(
    provider_name: str, model_calls: int, concurrency: int | None = None
) -> str:
    """``uses your Claude Code login, about 14 model calls, 4 at a time``."""
    parallel = _parallel(provider_name, concurrency)
    calls = "call" if model_calls == 1 else "calls"
    return (
        f"{PROVIDER_SPECS[provider_name].note}, about {model_calls:,} model {calls}, "
        f"{parallel} at a time"
    )


def subscription_estimate(
    provider_name: str, model_calls: int, concurrency: int | None = None
) -> str:
    """``uses your Claude Code login, about 14 model calls, 4 at a time, about 4 min``.

    What a subscription run costs is quota and time, not dollars, so this is
    what stands where an API provider shows its price. *concurrency* is the
    run's ``--concurrency``; the CLI's own process limit caps it.
    """
    minutes = subscription_minutes(provider_name, model_calls, concurrency)
    return f"{subscription_quota(provider_name, model_calls, concurrency)}, about {minutes:,} min"


def subscription_note(ready: tuple[str, ...]) -> str | None:
    """One line naming the first ready subscription CLI and the command that uses it.

    For a run that writes no documentation with a model: it never switches
    the run to the CLI, it says how to.
    """
    if not ready:
        return None
    name = ready[0]
    agent = identity_for_provider(name)
    if agent is None:
        return None
    label = agent.display_name.removesuffix(" CLI")
    state = "is logged in" if agent.login_check else "is installed"
    return (
        f"{label} {state}. Run [bold]repowise update --full --provider {name}[/bold] "
        f"to have it write the documentation ({PROVIDER_SPECS[name].note})."
    )


__all__ = [
    "AGENT_PROVIDERS",
    "SECONDS_PER_CALL",
    "SubscriptionContext",
    "agent_cli_status",
    "model_call_count",
    "process_limit",
    "ready_agent_providers",
    "subscription_estimate",
    "subscription_minutes",
    "subscription_note",
    "subscription_quota",
]
