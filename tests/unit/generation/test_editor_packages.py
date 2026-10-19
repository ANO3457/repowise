"""The editor files' package list: one row per package, from the graph alone."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from repowise.core.generation.editor_files.data import PackageRow
from repowise.core.generation.editor_files.fetcher import EditorFileDataFetcher
from repowise.core.generation.editor_files.packages import build_package_rows
from repowise.core.persistence.crud import upsert_repository
from repowise.core.persistence.database import init_db
from repowise.core.persistence.models import ExternalSystem, GraphNode, Page


def _files(lang: str, *paths: str, is_test: bool = False) -> list[tuple[str, str, bool]]:
    return [(p, lang, is_test) for p in paths]


def _page(path: str, summary: str, *, provider: str = "openai", meta: dict | None = None):
    return SimpleNamespace(
        target_path=path,
        summary=summary,
        content="",
        provider_name=provider,
        metadata_json=json.dumps(meta or {}),
    )


def _write(root: Path, rel: str, text: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")


def _paths(rows: list[PackageRow]) -> list[str]:
    return [r.path for r in rows]


def test_python_monorepo_has_one_row_per_package(tmp_path):
    _write(tmp_path, "packages/core/pyproject.toml", '[project]\nname = "acme-core"\n')
    _write(
        tmp_path,
        "packages/cli/pyproject.toml",
        '[project]\nname = "cli"\ndescription = "Command line front end. Ships the binary."\n',
    )
    files = [
        *_files("python", *(f"packages/core/src/acme/core/m{i}.py" for i in range(6))),
        *_files("python", "packages/cli/src/acme/cli/main.py", "packages/cli/src/acme/cli/a.py"),
        *_files("python", "packages/core/tests/test_m.py", is_test=True),
        *_files("python", "noxfile.py"),
        *_files("toml", "pyproject.toml", "packages/core/pyproject.toml"),
        *_files("toml", "packages/cli/pyproject.toml"),
    ]
    rows, omitted = build_package_rows(
        files, [], [], ["packages/cli/src/acme/cli/main.py"], repo_root=tmp_path
    )

    assert omitted == 0
    assert rows == [
        PackageRow(
            path="packages/cli",
            role="Command line front end",
            language="python",
            file_count=2,
            entry_point="packages/cli/src/acme/cli/main.py",
        ),
        PackageRow(
            path="packages/core",
            role="",
            language="python",
            file_count=6,
            entry_point=None,
            name="acme-core",
        ),
    ]


def test_single_python_package_opens_into_its_source_directories():
    files = [
        *_files("python", *(f"src/app/api/r{i}.py" for i in range(5))),
        *_files("python", *(f"src/app/db/m{i}.py" for i in range(4))),
        *_files("python", "src/app/cli/main.py", "src/app/__init__.py"),
        *_files("python", "tests/test_api.py", is_test=True),
        *_files("python", "docs/conf.py", "scripts/release.py", "examples/demo.py"),
        *_files("toml", "pyproject.toml"),
    ]
    rows, omitted = build_package_rows(files, ["pyproject.toml"], [], ["src/app/cli/main.py"])

    assert _paths(rows) == ["src/app", "src/app/api", "src/app/cli", "src/app/db"]
    assert [r.file_count for r in rows] == [1, 5, 1, 4]
    assert rows[2].entry_point == "src/app/cli/main.py"
    assert omitted == 0


def test_a_package_whose_code_sits_beside_its_manifest_stays_one_row():
    files = [*_files("go", *(f"m{i}.go" for i in range(8))), *_files("go", "doc/x.go")]
    rows, _ = build_package_rows(files, ["go.mod"], [], [])

    assert _paths(rows) == ["."]
    assert rows[0].language == "go"


def test_js_workspace_reads_names_and_descriptions_from_package_json(tmp_path):
    _write(
        tmp_path,
        "packages/web/package.json",
        json.dumps({"name": "@acme/web", "description": "Browser dashboard"}),
    )
    _write(tmp_path, "packages/api/package.json", json.dumps({"name": "api"}))
    files = [
        *_files("typescript", *(f"packages/web/src/c{i}.tsx" for i in range(3))),
        *_files("typescript", "packages/api/src/index.ts", "packages/api/src/routes.ts"),
        *_files("json", "package.json", "packages/web/package.json", "packages/api/package.json"),
        *_files("typescript", "packages/web/src/c.test.tsx", is_test=True),
    ]
    rows, _ = build_package_rows(
        files, ["packages/web/package.json"], [], ["packages/api/src/index.ts"], repo_root=tmp_path
    )

    assert rows == [
        PackageRow("packages/api", "", "typescript", 2, "packages/api/src/index.ts"),
        PackageRow("packages/web", "Browser dashboard", "typescript", 3, None, "@acme/web"),
    ]


def test_mixed_go_and_vue_repo():
    files = [
        *_files("go", "server/main.go", "server/api/h.go", "server/api/g.go"),
        *_files("vue", "web/src/App.vue", "web/src/main.vue"),
        *_files("javascript", "web/src/main.js"),
        *_files("json", "web/package.json"),
    ]
    rows, _ = build_package_rows(
        files, ["server/go.mod"], [], ["server/main.go", "web/src/main.js"]
    )

    assert [(r.path, r.language, r.entry_point) for r in rows] == [
        ("server", "go", "server/main.go"),
        ("web", "vue", "web/src/main.js"),
    ]


def test_rust_workspace_crates(tmp_path):
    _write(tmp_path, "crates/parse/Cargo.toml", '[package]\nname = "acme-parse"\n')
    _write(
        tmp_path,
        "crates/cli/Cargo.toml",
        '[package]\nname = "cli"\ndescription = "Command-line driver"\n',
    )
    files = [
        *_files("rust", "crates/parse/src/lib.rs", "crates/parse/src/lexer.rs"),
        *_files("rust", "crates/cli/src/main.rs"),
        *_files("toml", "Cargo.toml", "crates/parse/Cargo.toml", "crates/cli/Cargo.toml"),
    ]
    rows, _ = build_package_rows(files, [], [], [], repo_root=tmp_path)

    assert [(r.path, r.role, r.name) for r in rows] == [
        ("crates/cli", "Command-line driver", ""),
        ("crates/parse", "", "acme-parse"),
    ]


def test_model_prose_names_the_role_and_template_prose_never_does(tmp_path):
    _write(
        tmp_path,
        "packages/a/package.json",
        json.dumps({"name": "a", "description": "From the manifest"}),
    )
    _write(tmp_path, "packages/b/package.json", json.dumps({"name": "b"}))
    files = [
        *_files("typescript", *(f"packages/a/src/x{i}.ts" for i in range(4))),
        *_files("typescript", "packages/a/vitest.config.ts"),
        *_files("typescript", *(f"packages/b/src/y{i}.ts" for i in range(3))),
        *_files("json", "packages/a/package.json", "packages/b/package.json"),
    ]
    pages = [
        # The package root's own page covers the config beside the manifest.
        _page("packages/a", "The vitest config sets up the test runner."),
        _page("packages/a/src", "The **client** turns requests into typed calls. It retries."),
        _page("packages/b/src", "Package b holds y modules.", provider="template"),
        _page("packages/b", "Structure only.", meta={"deterministic": True}),
    ]
    rows, _ = build_package_rows(files, [], pages, [], repo_root=tmp_path)

    assert [r.role for r in rows] == ["The client turns requests into typed calls", ""]

    keyless = [p for p in pages if p.target_path.startswith("packages/b")]
    rows, _ = build_package_rows(files, [], keyless, [], repo_root=tmp_path)
    assert [r.role for r in rows] == ["From the manifest", ""]


def test_long_role_is_cut_to_one_clause():
    sentence = (
        "The orchestration layer turns repository files and language specifications "
        "into graphs, health findings, and wiki pages for every surface."
    )
    files = _files("python", "pkg/a.py", "pkg/sub/b.py", "other/c.py")
    rows, _ = build_package_rows(files, [], [_page("pkg", sentence)], [])

    role = rows[1].role
    assert len(role) <= 100 and not role.endswith(".")
    assert role.startswith("The orchestration layer turns repository files")


def test_no_manifest_falls_back_to_top_level_directories():
    files = [
        *_files("c", "src/a.c", "src/b.c", "src/c.c"),
        *_files("c", "lib/x.c", "lib/y.c"),
        *_files("c", "main.c"),
        *_files("c", "test/t.c", is_test=True),
    ]
    rows, omitted = build_package_rows(files, [], [], ["main.c", "src/a.c"])

    assert _paths(rows) == ["lib", "src"]
    assert rows[1].entry_point == "src/a.c"
    assert omitted == 0


def test_cap_keeps_the_largest_and_counts_the_rest():
    files = []
    for i in range(15):
        files += _files("go", *(f"svc{i:02d}/f{j}.go" for j in range(i + 1)))
    manifests = [f"svc{i:02d}/go.mod" for i in range(15)]
    rows, omitted = build_package_rows(files, manifests, [], [], cap=12)

    assert omitted == 3
    assert _paths(rows) == [f"svc{i:02d}" for i in range(3, 15)]


def test_tiny_directories_drop_but_tiny_packages_stay():
    files = [
        *_files("typescript", *(f"src/f{i}.ts" for i in range(100))),
        *_files("javascript", "bin/cli.js", "electron/main.js"),
        *_files("json", "electron/package.json"),
    ]
    rows, omitted = build_package_rows(files, ["package.json"], [], [])

    assert _paths(rows) == ["electron", "src"]
    assert omitted == 1


def test_hidden_and_tooling_directories_are_not_rows():
    files = [
        *_files("python", *(f"app/m{i}.py" for i in range(5))),
        *_files("shell", ".github/run.sh", "docker/entry.sh", "docs_src/tutorial/a.py"),
    ]
    rows, _ = build_package_rows(files, [], [], [])

    assert _paths(rows) == ["app"]


def test_output_does_not_depend_on_input_order():
    files = [
        *_files("python", *(f"packages/p{i}/src/m{j}.py" for i in range(5) for j in range(i + 2))),
        *_files("toml", *(f"packages/p{i}/pyproject.toml" for i in range(5))),
    ]
    pages = [_page(f"packages/p{i}/src", f"Package {i} does its own thing.") for i in range(5)]
    entry = ["packages/p3/src/m0.py", "packages/p1/src/m1.py"]
    expected = build_package_rows(files, [], pages, entry, cap=3)
    for seed in range(5):
        rng = random.Random(seed)
        shuffled_files, shuffled_pages = files[:], pages[:]
        rng.shuffle(shuffled_files)
        rng.shuffle(shuffled_pages)
        assert build_package_rows(shuffled_files, [], shuffled_pages, entry, cap=3) == expected
    assert expected[1] == 2


def test_empty_index_has_no_rows():
    assert build_package_rows([], [], [], []) == ([], 0)


# ---------------------------------------------------------------------------
# Fetcher: reads the graph, manifests and module pages from the store
# ---------------------------------------------------------------------------


@pytest.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    await init_db(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as sess:
        yield sess
    await engine.dispose()


async def test_fetcher_builds_rows_from_the_store(session, tmp_path):
    repo = await upsert_repository(session, name="acme", local_path=str(tmp_path), url="")
    _write(
        tmp_path,
        "services/api/pyproject.toml",
        '[project]\nname = "acme-api"\ndescription = "HTTP service"\n',
    )
    nodes = [
        ("services/api/src/acme_api/app.py", "python", False),
        ("services/api/src/acme_api/routes.py", "python", False),
        ("services/api/tests/test_app.py", "python", True),
        ("services/api/README.md", "markdown", False),
        ("web/src/index.ts", "typescript", False),
        ("web/package.json", "json", False),
        ("external:requests", "external", False),
    ]
    for node_id, language, is_test in nodes:
        session.add(
            GraphNode(
                repository_id=repo.id,
                node_id=node_id,
                node_type="file",
                language=language,
                is_test=is_test,
                is_entry_point=node_id == "web/src/index.ts",
            )
        )
    session.add(
        ExternalSystem(
            repository_id=repo.id,
            name="fastapi",
            ecosystem="pypi",
            declared_in="services/api/pyproject.toml",
        )
    )
    session.add(
        Page(
            id="module_page:web/src",
            repository_id=repo.id,
            page_type="module_page",
            title="Web",
            content="# Web\n\nThe browser client renders the dashboard.",
            summary="",
            target_path="web/src",
            source_hash="x",
            model_name="m",
            provider_name="openai",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    await session.flush()

    fetcher = EditorFileDataFetcher(session, repo.id, tmp_path)
    rows, omitted = await fetcher._get_packages()

    assert omitted == 0
    assert rows == [
        PackageRow("services/api", "HTTP service", "python", 2, None, "acme-api"),
        PackageRow("web", "The browser client renders the dashboard", "typescript", 1, "web/src/index.ts"),
    ]
    data = await fetcher.fetch()
    assert data.packages == rows and data.packages_omitted == 0


def test_rows_too_small_to_show_do_not_stop_the_opening():
    # One root package: its two shown directories are not enough to orient by,
    # and the tiny ones that would make four rows are left out anyway.
    files = [
        *_files("python", *(f"app/api/m{i}.py" for i in range(40))),
        *_files("python", *(f"app/front/m{i}.py" for i in range(30))),
        *_files("javascript", *(f"static/js/m{i}.js" for i in range(10))),
        *_files("python", "tools_a/x.py", "tools_b/y.py"),
        *_files("text", "requirements.txt"),
    ]
    rows, omitted = build_package_rows(files, ["requirements.txt"], [], [])

    assert _paths(rows) == ["app/api", "app/front", "static/js"]
    assert omitted == 2

def test_a_localisation_key_is_not_a_role(tmp_path):
    _write(tmp_path, "ext/package.json", '{"name": "ext", "description": "%extension.description%"}')
    files = [
        *_files("typescript", "ext/src/a.ts", "ext/src/b.ts"),
        *_files("typescript", "lib/src/c.ts"),
        *_files("json", "ext/package.json", "lib/package.json"),
    ]
    manifests = ["ext/package.json", "lib/package.json"]
    rows, _ = build_package_rows(files, manifests, [], [], repo_root=tmp_path)

    assert rows[0].path == "ext" and rows[0].role == ""