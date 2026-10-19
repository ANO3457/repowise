"""One naming set for index modes across init, status and update output."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from repowise.cli.ui import mode_names as names
from repowise.core.index_scope import resolve_index_scope

_CLI = Path(names.__file__).resolve().parents[1]


def _scope(docs_mode: str | None, **state):
    if docs_mode is not None:
        state["docs_mode"] = docs_mode
    return resolve_index_scope(state, {"commit_limit": 5000})


class TestTable:
    @pytest.mark.parametrize(
        ("docs_mode", "model", "label"),
        [
            ("deterministic", None, "Built from code"),
            ("llm", "gemini-2.5-flash", "Written by gemini-2.5-flash"),
            ("llm", None, "Written by a model"),
            ("none", None, "Index only"),
            (None, None, "Mode unknown"),
        ],
    )
    def test_docs_mode_label(self, docs_mode, model, label):
        assert names.docs_mode_label(docs_mode, model) == label

    def test_mode_label_reads_the_scope_model(self):
        scope = _scope("llm", model="mock-model")
        assert names.mode_label(scope) == "Written by mock-model"

    def test_fast_with_unknown_provenance_is_index_only(self):
        assert names.mode_label(_scope(None, run_mode="fast")) == "Index only"

    @pytest.mark.parametrize(
        ("docs_mode", "title"),
        [
            ("deterministic", "Index ready, documentation built from code"),
            ("llm", "Repowise is ready"),
            ("none", "Index only ready"),
        ],
    )
    def test_completion_title(self, docs_mode, title):
        assert names.completion_title(_scope(docs_mode)) == title

    def test_scope_line_carries_mode_and_commit_cap(self):
        scope = _scope("deterministic", git_tier="full")
        assert names.scope_line(scope) == "Built from code · git full · last 5,000 commits"

    def test_scope_line_without_a_recorded_cap(self):
        scope = resolve_index_scope({"docs_mode": "none", "git_tier": "essential"}, {})
        assert names.scope_line(scope) == "Index only · git essential"

    def test_commit_cap_singular(self):
        assert names.commit_cap_label(1) == "last 1 commit"


# The init, status and update modules whose output names a mode.
_SURFACES = [
    "ui/mode_names.py",
    "ui/mode_selection.py",
    "ui/result_panels.py",
    "commands/status_cmd.py",
    "commands/init_cmd/command.py",
    "commands/init_cmd/reporting.py",
    "commands/init_cmd/_provider.py",
    "commands/init_cmd/_generate.py",
    "commands/init_cmd/_prepare.py",
    "commands/init_cmd/_options.py",
    "commands/init_cmd/generation.py",
    "commands/init_cmd/workspace.py",
    "commands/update_cmd/reporting.py",
]

_BANNED = [
    re.compile(r"\bNo prose\b", re.I),
    re.compile(r"(?<!--)\bindex-only\b", re.I),
    re.compile(r"\bwiki pages?\b", re.I),
    re.compile(r"\bEverything\b"),
    re.compile(r"\btemplates?\b", re.I),
    re.compile(r"\bmodel-written\b", re.I),
    re.compile(r"\bfrom structure\b", re.I),
    # An em dash in prose; a lone one is the empty-cell placeholder.
    re.compile("\\s\u2014|\u2014\\s"),
]

# State and provider values compared in code, never shown.
_IDENTIFIER = re.compile(r"[a-z_]+")


def _user_strings(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and not _IDENTIFIER.fullmatch(node.value)
        ):
            yield node.lineno, node.value


@pytest.mark.parametrize("relative", _SURFACES)
def test_no_retired_mode_names_in_user_strings(relative):
    hits = [
        f"{relative}:{line}: {text!r}"
        for line, text in _user_strings(_CLI / relative)
        for pattern in _BANNED
        if pattern.search(text)
    ]
    assert not hits, "\n".join(hits)
