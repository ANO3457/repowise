"""The editor files' package list: one row per package, built from the graph.

Rows are the container roots the C4 view uses (a directory holding a package
manifest). A repository with one root is opened into its source directories
instead, the largest first, until there are enough rows to orient by, so the
list is never a lone ``.``. In a monorepo the root catch-all is opened into its
top-level directories the same way, since files no package claims are not a
package. Only production code counts: tests, examples, docs, build tooling and
non-code files neither make a row nor add to its ``file_count``.

A row's role is the first clause of the module page a model wrote for the
directory holding most of its code, or for the nearest directory above that one
within the row; else the description its manifest declares; else ``""``.
Template pages restate the structure the row already shows, so they never
supply one.

Rows are ordered by path. Past ``cap`` the smallest rows are left out, as are
directories too small to orient by, and both count in the omitted total.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from repowise.core.analysis.c4.containers import assign_containers, container_name, is_manifest
from repowise.core.analysis.kg_inputs import _dominant_language
from repowise.core.generation.entry_points import TOOLING_DIRS
from repowise.core.generation.layers import majority_root
from repowise.core.ingestion.languages.registry import REGISTRY
from repowise.core.ingestion.workspace_members import manifest_description
from repowise.core.support_paths import is_support_path, is_test_or_example_path

from .data import PackageRow
from .fetcher import _extract_sentences, _truncate_at_word

PACKAGE_CAP = 12
#: A single-package repo is opened until it has at least this many rows.
_ROW_FLOOR = 4
_ROLE_CHARS = 100
#: A directory holding less of the production code than this is not shown,
#: unless a manifest makes it a package.
_MINOR_SHARE = 0.02

CODE_LANGUAGES = REGISTRY.code_languages()
_PLACEHOLDER = re.compile(r"%[\w.-]+%")


def build_package_rows(
    files: Iterable[tuple[str, str | None, bool]],
    manifests: Iterable[str],
    module_pages: Iterable[Any],
    entry_points: list[str],
    *,
    repo_root: Path | None = None,
    cap: int = PACKAGE_CAP,
) -> tuple[list[PackageRow], int]:
    """``(rows, omitted)`` for the package list.

    *files* are ``(path, language, is_test)`` file nodes; *manifests* the
    ``declared_in`` paths of external dependencies; *module_pages* carry
    ``target_path``, ``summary``, ``content``, ``provider_name`` and
    ``metadata_json``; *entry_points* is the curated orientation list, best
    first. *repo_root* lets manifests be read for names and descriptions.
    """
    files = list(files)
    support: dict[str, bool] = {}
    language = {
        path: lang
        for path, lang, is_test in files
        if not is_test and lang in CODE_LANGUAGES and not _is_support_dir(path, support)
    }
    owner, root_manifests = assign_containers(
        manifests, [*language, *(p for p, _, _ in files if is_manifest(p))]
    )
    heads: dict[str, list[str]] = defaultdict(list)
    for path in language:
        if path in owner:
            heads[owner[path]].append(path)
    floor = len(language) * _MINOR_SHARE
    heads, opened = _widened(dict(heads), floor)
    if not heads:
        return [], 0
    # An opened root's row holds only the files beside its manifest, not the package.
    packages = {h for h in heads if h in root_manifests and h not in opened}

    by_size = sorted(heads, key=lambda h: (-len(heads[h]), h))
    kept = [h for h in by_size if h in packages or len(heads[h]) >= floor][:cap]
    kept = kept or by_size[:1]
    prose = _page_roles(module_pages)
    rows = []
    for head in sorted(kept):
        manifests_here = root_manifests.get(head, ()) if head in packages else ()
        name = (container_name(repo_root, manifests_here) or "") if manifests_here else ""
        own = set(heads[head])
        rows.append(
            PackageRow(
                path=head or ".",
                role=_role(head, heads[head], prose, repo_root, manifests_here),
                language=_dominant_language([language[p] for p in heads[head]]),
                file_count=len(heads[head]),
                entry_point=next((e for e in entry_points if e in own), None),
                name=name if name.lower() != head.rsplit("/", 1)[-1].lower() else "",
            )
        )
    return rows, len(heads) - len(kept)


def _widened(
    heads: dict[str, list[str]], floor: float
) -> tuple[dict[str, list[str]], set[str]]:
    """Open a lone root, or a monorepo's root catch-all, into its subdirectories.

    Files directly at the root of a monorepo are build glue and are dropped.
    Then, while fewer than ``_ROW_FLOOR`` rows clear *floor*, the largest row that
    is not itself a package keeps opening; an opened directory stays a row for
    the files directly in it. Returns the rows and the directories that were opened.
    """
    opened: set[str] = set()
    packages: set[str] = set()
    if len(heads) > 1:
        packages = set(heads) - {""}
        if "" in heads:
            heads.update(_children("", heads.pop("")))
    # Rows below *floor* are not shown, so they do not count toward the floor.
    while sum(len(paths) >= floor for paths in heads.values()) < _ROW_FLOOR:
        openable = {
            h: c
            for h in heads
            if h not in opened and h not in packages and (c := _children(h, heads[h]))
        }
        if not openable:
            break
        widest = max(openable, key=lambda h: (len(heads[h]), h))
        moved = {p for paths in openable[widest].values() for p in paths}
        if direct := [p for p in heads.pop(widest) if p not in moved]:
            heads[widest] = direct
        heads.update(openable[widest])
        opened.add(widest)
    return heads, opened


def _children(head: str, paths: list[str]) -> dict[str, list[str]]:
    """*paths* grouped by the subdirectory of *head* holding them; direct files drop."""
    out: dict[str, list[str]] = defaultdict(list)
    prefix = f"{head}/" if head else ""
    for path in paths:
        rest = path[len(prefix) :]
        if "/" in rest:
            out[prefix + rest.split("/", 1)[0]].append(path)
    return dict(out)


def _is_support_dir(path: str, cache: dict[str, bool]) -> bool:
    """Whether *path* sits in tests, docs, examples, tooling or a hidden directory.

    Doc trees include prefixed names (``docs_src``, ``docs-site``), whose code
    is runnable snippets, not the system.

    Cached per directory. The test check backs up the stored ``is_test`` flag,
    which an index older than its classifier can have left unset.
    """
    directory = path.rsplit("/", 1)[0] if "/" in path else ""
    if directory not in cache:
        segments = directory.lower().split("/") if directory else []
        stand_in = f"{directory}/_"  # the basename is ignored, so any name will do
        cache[directory] = (
            any(s.startswith((".", "docs_", "docs-")) or s in TOOLING_DIRS for s in segments)
            or is_support_path(stand_in)
            or is_test_or_example_path(stand_in)
        )
    return cache[directory]


def _under(path: str, root: str) -> bool:
    return not root or path == root or path.startswith(root + "/")


def _page_roles(module_pages: Iterable[Any]) -> dict[str, str]:
    """``{target_path: role}`` for module pages a model wrote."""
    roles: dict[str, str] = {}
    for page in module_pages:
        if _is_template(page):
            continue
        text = page.summary or page.content or ""
        path = "" if page.target_path in (None, ".") else page.target_path
        roles[path] = _clause(_extract_sentences(text, max_sentences=1))
    return roles


def _is_template(page: Any) -> bool:
    if page.provider_name == "template":
        return True
    try:
        meta = json.loads(page.metadata_json or "{}")
    except (TypeError, ValueError):
        return False
    return isinstance(meta, dict) and bool(meta.get("deterministic") or meta.get("model_free"))


def _role(
    head: str, paths: list[str], prose: dict[str, str], repo_root: Path | None, manifests: Any
) -> str:
    # The page for the directory holding most of the row's code speaks for it:
    # a package root's own page often covers only the config files beside it.
    directory = majority_root(paths) or ""
    while _under(directory, head):
        if prose.get(directory):
            return prose[directory]
        if directory == head:
            break
        directory = directory.rsplit("/", 1)[0] if "/" in directory else ""
    if repo_root is None:
        return ""
    for manifest in manifests:
        text = manifest_description(repo_root / manifest)
        # A localisation key (``%extension.description%``) is not a description.
        if text and not _PLACEHOLDER.fullmatch(text.strip()):
            return _clause(text.split(". ", 1)[0])
    return ""


def _clause(sentence: str) -> str:
    """*sentence* cut to one clause of at most ``_ROLE_CHARS``, no trailing period."""
    sentence = sentence.replace("**", "").strip().rstrip(".")
    if len(sentence) <= _ROLE_CHARS:
        return sentence
    head = sentence[:_ROLE_CHARS]
    cut = max(head.rfind(", "), head.rfind("; "), head.rfind(" — "), head.rfind(": "))
    if cut >= _ROLE_CHARS // 3:
        return head[:cut]
    return _truncate_at_word(sentence, _ROLE_CHARS)
