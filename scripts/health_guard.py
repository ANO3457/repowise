"""Advisory PR guard: touched files keep their code shape, no new oversized function.

Compares both sides of a change in memory (no index needed) and prints GitHub
warning annotations plus a job-summary table. It never fails the build.

    python scripts/health_guard.py --revspec origin/main...HEAD
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

from repowise.core.analysis.change_health.analyzer import (
    RevisionAnalysis,
    RevisionHealthAnalyzer,
    language_for,
)
from repowise.core.analysis.change_health.service import MAX_CHANGED_FILES
from repowise.core.analysis.change_health.sources import (
    FileChange,
    GitRevisionSource,
    RevisionPair,
)
from repowise.core.analysis.health.complexity.walker import walk_file_complexity
from repowise.core.analysis.health.counts import code_shape_score
from repowise.core.analysis.health.scoring import deduction_split
from repowise.core.ci.github import annotation, cap_annotations, notice
from repowise.core.ci.markdown import cell

MAX_CCN = 50
MAX_NLOC = 300
_DROP_EPSILON = 0.005


@dataclass(frozen=True, slots=True)
class ShapeDrop:
    path: str
    before: float
    after: float


@dataclass(frozen=True, slots=True)
class LargeFunction:
    path: str
    name: str
    line: int
    ccn: int
    nloc: int


@dataclass(slots=True)
class GuardReport:
    drops: list[ShapeDrop] = field(default_factory=list)
    functions: list[LargeFunction] = field(default_factory=list)
    note: str = ""


def _oversized(ccn: int, nloc: int) -> bool:
    return ccn > MAX_CCN or nloc > MAX_NLOC


def _shape(run: RevisionAnalysis, path: str) -> float | None:
    if path not in run.analyzed:
        return None
    structure, _ = deduction_split(run.findings_for({path}))
    return code_shape_score(structure)


def _functions(path: str, source: bytes | None) -> list:
    language = language_for(path)
    if source is None or language is None:
        return []
    return walk_file_complexity(path, language, source)


def new_large_functions(path: str, base: bytes | None, head: bytes) -> list[LargeFunction]:
    """Head functions over a limit beyond the base's count of over-limit functions so named."""
    budget = Counter(fn.name for fn in _functions(path, base) if _oversized(fn.ccn, fn.nloc))
    out: list[LargeFunction] = []
    for fn in _functions(path, head):
        if not _oversized(fn.ccn, fn.nloc):
            continue
        if budget[fn.name] > 0:
            budget[fn.name] -= 1
            continue
        out.append(LargeFunction(path, fn.name, fn.start_line, fn.ccn, fn.nloc))
    return out


class _Sides(NamedTuple):
    base_src: dict[str, bytes]
    head_src: dict[str, bytes]
    base_run: RevisionAnalysis
    head_run: RevisionAnalysis


def _analyze(source: GitRevisionSource, pair: RevisionPair, changes: list[FileChange]) -> _Sides:
    head_paths = [c.head_path for c in changes]
    base_paths = [c.base_path for c in changes if c.base_path]
    head_src = source.read(pair.head_sha, head_paths)
    base_src = source.read(pair.base_sha, base_paths)
    analyzer = RevisionHealthAnalyzer(repo_root=source.repo_path)
    head_run = analyzer.analyze(head_src, subject_paths=set(head_paths))
    to_base = {c.head_path: c.base_path for c in changes if c.base_path}
    # Score base against head's dependents so a centrality shift is not read as a code change.
    on_base = {to_base.get(p, p): n for p, n in head_run.dependents.items()}
    base_run = analyzer.analyze(base_src, subject_paths=set(base_paths), dependents=on_base)
    return _Sides(base_src, head_src, base_run, head_run)


def check(repo_path: str, revspec: str) -> GuardReport:
    """Compare *revspec*'s two sides for the touched files health can read."""
    source = GitRevisionSource(repo_path, hunks=False)
    pair = source.resolve(revspec)
    changes = [c for c in pair.changes if c.head_path and language_for(c.head_path)]
    if len(changes) > MAX_CHANGED_FILES:
        note = f"{len(changes)} changed files; the guard reads at most {MAX_CHANGED_FILES}."
        return GuardReport(note=note)
    if not changes:
        return GuardReport()
    sides = _analyze(source, pair, changes)
    report = GuardReport()
    for change in changes:
        head, base = change.head_path, change.base_path
        # Generated, binary and oversized files never reach the analyzer; skip them here too.
        if head not in sides.head_run.analyzed:
            continue
        base_bytes = sides.base_src.get(base) if base else None
        report.functions += new_large_functions(head, base_bytes, sides.head_src[head])
        before = _shape(sides.base_run, base) if base else None
        after = _shape(sides.head_run, head)
        if before is not None and after is not None and after < before - _DROP_EPSILON:
            report.drops.append(ShapeDrop(head, before, after))
    return report


def annotations(report: GuardReport) -> list[str]:
    """One warning per drop and per function newly over a limit, capped like every CI gate."""
    lines = [
        annotation(
            "warning",
            f"Code shape {d.before:.2f} -> {d.after:.2f}. "
            "Keep touched files at or above their base score.",
            file=d.path,
            title="Code shape dropped",
        )
        for d in report.drops
    ]
    lines += [
        annotation(
            "warning",
            f"{f.name}: CCN {f.ccn}, {f.nloc} code lines "
            f"(limits {MAX_CCN} / {MAX_NLOC}). Split it before merging.",
            file=f.path,
            line=f.line,
            title="Function over the size limit",
        )
        for f in report.functions
    ]
    if report.note:
        lines.append(notice(report.note))
    return cap_annotations(lines, noun="health guard warnings")


def summary(report: GuardReport) -> str:
    """Markdown for the job summary; empty when there is nothing to say."""
    if not (report.drops or report.functions or report.note):
        return ""
    rows = ["### Health guard (advisory)", ""]
    if report.note:
        rows += [report.note, ""]
    if report.drops:
        rows += ["| File | Code shape before | After |", "|---|---:|---:|"]
        rows += [f"| {cell(d.path)} | {d.before:.2f} | {d.after:.2f} |" for d in report.drops]
        rows.append("")
    if report.functions:
        rows += ["| Function | File | CCN | Code lines |", "|---|---|---:|---:|"]
        rows += [
            f"| {cell(f.name)} | {cell(f.path)}:{f.line} | {f.ccn} | {f.nloc} |"
            for f in report.functions
        ]
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".")
    parser.add_argument("--revspec", default="origin/main...HEAD")
    args = parser.parse_args(argv)
    try:
        report = check(args.repo, args.revspec)
    except Exception as exc:  # advisory: a guard failure must never block the PR
        print(annotation("warning", f"Health guard did not run: {exc}"))
        return 0
    for line in annotations(report):
        print(line)
    text = summary(report)
    if text and (target := os.environ.get("GITHUB_STEP_SUMMARY")):
        try:
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            print(annotation("warning", f"Health guard could not write the job summary: {exc}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
