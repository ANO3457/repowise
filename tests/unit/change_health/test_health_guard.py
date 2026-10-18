"""The advisory PR guard: code-shape drops on touched files and new oversized functions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .conftest import python_complex

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "health_guard.py"


def _guard():
    spec = importlib.util.spec_from_file_location("health_guard", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["health_guard"] = module
    spec.loader.exec_module(module)
    return module


def _long(name: str, lines: int) -> str:
    body = "".join(f"    total = total + {i}\n" for i in range(lines))
    return f"def {name}(total):\n{body}    return total\n"


def test_a_touched_file_that_loses_code_shape_is_reported(make_repo):
    guard = _guard()
    repo = make_repo()
    base = repo.commit("seed", {"app.py": python_complex("small", 2)})
    repo.commit("grow", {"app.py": python_complex("small", 2) + python_complex("tangle", 30)})

    report = guard.check(str(repo.path), f"{base}..HEAD")

    assert [d.path for d in report.drops] == ["app.py"]
    assert report.drops[0].after < report.drops[0].before


def test_a_new_function_over_the_ccn_or_line_limit_is_reported(make_repo):
    guard = _guard()
    repo = make_repo()
    base = repo.commit("seed", {"app.py": python_complex("small", 2)})
    repo.commit(
        "add",
        {
            "app.py": python_complex("small", 2) + python_complex("tangle", 30),
            "long.py": _long("sprawl", 320),
        },
    )

    report = guard.check(str(repo.path), f"{base}..HEAD")

    flagged = {(f.path, f.name) for f in report.functions}
    assert flagged == {("app.py", "tangle"), ("long.py", "sprawl")}


def test_a_function_already_over_the_limit_is_not_new(make_repo):
    guard = _guard()
    repo = make_repo()
    base = repo.commit("seed", {"app.py": python_complex("tangle", 30)})
    repo.commit("trim", {"app.py": python_complex("tangle", 28)})

    report = guard.check(str(repo.path), f"{base}..HEAD")

    assert report.functions == []
    assert report.drops == []


def test_output_is_advisory_warnings_and_a_summary(make_repo, tmp_path, monkeypatch, capsys):
    guard = _guard()
    repo = make_repo()
    base = repo.commit("seed", {"app.py": python_complex("small", 2)})
    repo.commit("grow", {"app.py": python_complex("small", 2) + python_complex("tangle", 30)})
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    code = guard.main(["--repo", str(repo.path), "--revspec", f"{base}..HEAD"])

    out = capsys.readouterr().out
    assert code == 0
    assert "::warning file=app.py" in out
    assert "tangle" in summary.read_text(encoding="utf-8")


def test_a_clean_change_prints_nothing(make_repo, capsys):
    guard = _guard()
    repo = make_repo()
    base = repo.commit("seed", {"app.py": python_complex("small", 2)})
    repo.commit("docs", {"README.md": "hello\n"})

    assert guard.main(["--repo", str(repo.path), "--revspec", f"{base}..HEAD"]) == 0
    assert "::warning" not in capsys.readouterr().out


def test_a_second_over_limit_function_of_the_same_name_is_reported(make_repo):
    guard = _guard()
    repo = make_repo()
    one = "class A:\n" + python_complex("run", 30, indent="    ")
    two = one + "\n\nclass B:\n" + python_complex("run", 30, indent="    ")
    base = repo.commit("seed", {"app.py": one})
    repo.commit("add", {"app.py": two})

    report = guard.check(str(repo.path), f"{base}..HEAD")

    assert [f.name for f in report.functions] == ["run"]


def test_a_renamed_file_compares_against_its_base_path(make_repo):
    guard = _guard()
    repo = make_repo()
    base = repo.commit("seed", {"old.py": python_complex("tangle", 30)})
    repo.move("old.py", "new.py")
    repo.commit("rename")

    report = guard.check(str(repo.path), f"{base}..HEAD")

    assert report.functions == []
    assert report.drops == []


def test_a_revspec_that_does_not_resolve_warns_and_passes(make_repo, capsys):
    guard = _guard()
    repo = make_repo()
    repo.commit("seed", {"app.py": python_complex("small", 2)})

    assert guard.main(["--repo", str(repo.path), "--revspec", "nope...HEAD"]) == 0
    assert "::warning::Health guard did not run" in capsys.readouterr().out


def test_an_unwritable_summary_warns_and_passes(make_repo, tmp_path, monkeypatch, capsys):
    guard = _guard()
    repo = make_repo()
    base = repo.commit("seed", {"app.py": python_complex("small", 2)})
    repo.commit("grow", {"app.py": python_complex("small", 2) + python_complex("tangle", 30)})
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path))

    assert guard.main(["--repo", str(repo.path), "--revspec", f"{base}..HEAD"]) == 0
    assert "could not write the job summary" in capsys.readouterr().out
