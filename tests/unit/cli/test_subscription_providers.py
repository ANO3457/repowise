"""Own-subscription providers are detected and offered, never picked for the user."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from repowise.cli.commands.init_cmd import _provider
from repowise.cli.cost_gate import spend_phrase
from repowise.cli.ui import provider_selection as ui
from repowise.cli.ui import subscription
from repowise.cli.ui.brand import THEME
from repowise.core.cost_estimator import approximate_generation_plan, estimate_cost
from repowise.core.providers.llm import resolve
from repowise.core.providers.llm.specs import PROVIDER_SPECS

_KEYS = tuple(
    var for spec in PROVIDER_SPECS.values() for var in (*spec.api_key_envs, *spec.base_url_envs)
)


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: Any, tmp_path: Path) -> None:
    for var in (*_KEYS, "REPOWISE_PROVIDER", "REPOWISE_CLAUDE_CLI_CONCURRENCY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("REPOWISE_CONFIG_DIR", str(tmp_path / "global"))
    monkeypatch.setattr(ui, "_detect_ollama_status", lambda: False)
    resolve.clear_cache()


def _logged_in(monkeypatch: Any, *names: str) -> None:
    monkeypatch.setattr(subscription, "agent_cli_status", lambda name: (True, name in names))


def _default_pick(monkeypatch: Any, repo: Path | None = None, **kwargs: Any) -> tuple[str, str]:
    seen: dict[str, str] = {}

    def ask(*_a: object, default: str, **_k: object) -> str:
        seen["default"] = default
        return default

    monkeypatch.setattr(ui.Prompt, "ask", ask)
    buf = io.StringIO()
    ui._interactive_provider_name(Console(file=buf, width=160, theme=THEME), None, repo_path=repo, **kwargs)
    return ui._PROVIDER_CHOICES[int(seen["default"]) - 1], buf.getvalue()


# -- detection ----------------------------------------------------------------


def test_ready_agent_providers_lists_only_logged_in_clis(monkeypatch: Any) -> None:
    _logged_in(monkeypatch, "claude_cli")
    assert subscription.ready_agent_providers() == ("claude_cli",)
    _logged_in(monkeypatch)
    assert subscription.ready_agent_providers() == ()


def test_picker_and_note_share_one_detection(monkeypatch: Any) -> None:
    _logged_in(monkeypatch, "codex_cli", "claude_cli")
    detected = ui._detect_provider_status()
    assert [p for p in detected if PROVIDER_SPECS[p].agent] == list(
        subscription.ready_agent_providers()
    )


# -- picker: same keys and order as auto-detection ------------------------------


def test_picker_default_matches_autodetect_order(monkeypatch: Any) -> None:
    _logged_in(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert resolve.resolve_llm(None).provider == "anthropic"
    assert _default_pick(monkeypatch)[0] == "anthropic"


def test_picker_reads_repo_env_and_stored_keys(monkeypatch: Any, tmp_path: Path) -> None:
    _logged_in(monkeypatch)
    repo = tmp_path / "repo"
    (repo / ".repowise").mkdir(parents=True)
    (repo / ".repowise" / ".env").write_text("DEEPSEEK_API_KEY=d\n", encoding="utf-8")
    stored = tmp_path / "global" / "provider_config.json"
    stored.parent.mkdir(parents=True)
    stored.write_text(json.dumps({"keys": {"kimi": "k"}}), encoding="utf-8")

    detected = ui._detect_provider_status(repo)

    assert list(detected) == ["deepseek", "kimi"]
    assert _default_pick(monkeypatch, repo)[0] == "deepseek"


def test_a_key_outranks_a_logged_in_cli_without_a_preference(monkeypatch: Any) -> None:
    _logged_in(monkeypatch, "claude_cli")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert _default_pick(monkeypatch)[0] == "openai"


def test_picker_shows_quota_and_time_for_a_logged_in_cli(monkeypatch: Any) -> None:
    _logged_in(monkeypatch, "claude_cli")
    _chosen, out = _default_pick(
        monkeypatch, subscription=subscription.SubscriptionContext(model_calls=14, concurrency=10)
    )
    assert "claude_cli uses your Claude Code login, about 14 model calls, 4 at a time" in out
    assert "about 4 min" in out
    assert "$0.00" not in out


def test_picker_without_an_estimate_prints_no_quota_line(monkeypatch: Any) -> None:
    _logged_in(monkeypatch, "claude_cli")
    assert "model calls" not in _default_pick(monkeypatch)[1]


# -- the estimate ---------------------------------------------------------------


def test_estimate_divides_calls_by_the_process_limit(monkeypatch: Any) -> None:
    text = subscription.subscription_estimate("claude_cli", 40, 10)
    assert "about 40 model calls, 4 at a time, about 10 min" in text
    monkeypatch.setenv("REPOWISE_CLAUDE_CLI_CONCURRENCY", "8")
    assert "8 at a time, about 5 min" in subscription.subscription_estimate("claude_cli", 40, 10)
    # A lower --concurrency still lowers it.
    assert "2 at a time, about 20 min" in subscription.subscription_estimate("claude_cli", 40, 2)


def test_spend_phrase_prices_api_providers_and_estimates_subscriptions() -> None:
    plans = approximate_generation_plan(50)
    calls = subscription.model_call_count(plans)
    cli = spend_phrase(estimate_cost(plans, "claude_cli", "claude_cli/claude-haiku-5-5"), 10)
    assert cli.startswith("Uses your Claude Code login")
    assert f"about {calls} model calls" in cli
    assert "$" not in cli
    assert spend_phrase(estimate_cost(plans, "gemini", "gemini-3.5-flash-lite")).startswith(
        "Estimated"
    )


# -- the --yes note -------------------------------------------------------------


def _run(**overrides: Any) -> SimpleNamespace:
    opts = SimpleNamespace(provider_name=overrides.pop("provider_name", None), index_only=True)
    fields: dict[str, Any] = {"is_interactive": False, "config": {}, "no_provider": False}
    return SimpleNamespace(opts=opts, **{**fields, **overrides})


def _note(monkeypatch: Any, run: SimpleNamespace) -> str:
    buf = io.StringIO()
    monkeypatch.setattr(_provider, "console", Console(file=buf, width=200, theme=THEME))
    _provider.note_subscription_cli(run)
    return buf.getvalue()


def test_yes_run_names_the_logged_in_cli_and_stays_free(monkeypatch: Any) -> None:
    _logged_in(monkeypatch, "claude_cli")
    out = _note(monkeypatch, _run())
    assert "Claude Code is logged in" in out
    assert "repowise update --full --provider claude_cli" in out


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider_name": "gemini"},
        {"is_interactive": True},
        {"config": {"provider": "openai"}},
    ],
)
def test_note_is_quiet_when_the_user_already_chose(monkeypatch: Any, overrides: dict) -> None:
    _logged_in(monkeypatch, "claude_cli")
    assert _note(monkeypatch, _run(**overrides)) == ""


def test_note_is_quiet_when_a_model_writes_the_docs(monkeypatch: Any) -> None:
    _logged_in(monkeypatch, "claude_cli")
    run = _run()
    run.opts.index_only = False
    assert _note(monkeypatch, run) == ""


def test_note_is_quiet_without_a_logged_in_cli(monkeypatch: Any) -> None:
    _logged_in(monkeypatch)
    assert _note(monkeypatch, _run()) == ""


def test_opencode_note_claims_no_login(monkeypatch: Any) -> None:
    _logged_in(monkeypatch, "opencode")
    out = _note(monkeypatch, _run())
    assert "OpenCode is installed" in out
    assert "logged in" not in out
