"""Spring Cloud Stream bindings: YAML/properties bindings and StreamBridge sends.

A binding maps a function's logical input or output to a destination:
``bindings.chargeOrder-in-0.destination: orders``.

- ``-in-N`` bindings are consumers of the configured destination.
- ``-out-N`` bindings are providers of the configured destination.
- ``StreamBridge.send("bindingOrDest", payload)`` calls are providers.
- A binding without a destination produces no contract.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from repowise.core.workspace.contracts import TOPIC_KIND_TOPIC
from repowise.core.workspace.origins import ContractOrigin

from ..langs import JAVA, KOTLIN
from .dialect import topic_contract

if TYPE_CHECKING:
    from repowise.core.workspace.contracts import Contract

    from ..base import ScanContext

_BROKER = "cloud_stream"
_YAML_EXT = frozenset({".yaml", ".yml", ".properties"})
_EXTENSIONS = JAVA | KOTLIN | _YAML_EXT
_GATES = ("destination", "StreamBridge")

_STREAM_BRIDGE_SEND_RE = re.compile(
    r"\bstreamBridge\s*\.\s*send\s*\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

_BINDING_KEY_RE = re.compile(r"""^(\s*)([A-Za-z0-9_.-]+-(?:in|out)-\d+)\s*:""")
_DEST_KEY_RE = re.compile(r"""^\s*destination\s*[:=]\s*['"]?([A-Za-z0-9_.-]+)['"]?""")
_FLAT_RE = re.compile(
    r"""(?:\bbindings\.)?([A-Za-z0-9_.-]+-(?:in|out)-\d+)\.destination\s*[:=]\s*['"]?([A-Za-z0-9_.-]+)['"]?"""
)


class _CloudStreamPass:
    def __init__(self) -> None:
        self._binding_destinations: dict[str, str] = {}
        self._binding_rows: list[tuple[ScanContext, str, str, str, int]] = []
        self._send_calls: list[tuple[ScanContext, str, int]] = []

    def scan(self, ctx: ScanContext) -> None:
        content = ctx.content
        if ctx.suffix in _YAML_EXT:
            current_binding: tuple[str, int] | None = None
            current_indent = -1
            offset = 0
            for line in content.splitlines(keepends=True):
                m_flat = _FLAT_RE.search(line)
                if m_flat:
                    b_name = m_flat.group(1).split(".")[-1]
                    dest = m_flat.group(2)
                    self._binding_destinations[b_name] = dest
                    role = "consumer" if "-in-" in b_name else "provider"
                    self._binding_rows.append((ctx, b_name, dest, role, offset))
                    offset += len(line)
                    continue

                m_b = _BINDING_KEY_RE.search(line)
                if m_b:
                    current_indent = len(m_b.group(1))
                    current_binding = (m_b.group(2), offset)
                    offset += len(line)
                    continue

                if current_binding:
                    indent = len(line) - len(line.lstrip())
                    if indent > current_indent:
                        m_d = _DEST_KEY_RE.search(line)
                        if m_d:
                            b_name = current_binding[0]
                            dest = m_d.group(1)
                            self._binding_destinations[b_name] = dest
                            role = "consumer" if "-in-" in b_name else "provider"
                            self._binding_rows.append((ctx, b_name, dest, role, current_binding[1]))
                            current_binding = None
                    else:
                        current_binding = None
                offset += len(line)

        if "StreamBridge" in content:
            for m in _STREAM_BRIDGE_SEND_RE.finditer(content):
                target = m.group(1)
                self._send_calls.append((ctx, target, m.start()))

    def finish(self) -> list[Contract]:
        out: list[Contract] = []
        seen: set[tuple[str, str, str]] = set()

        for ctx, b_name, dest, role, offset in self._binding_rows:
            key = (dest.lower(), role)
            if key not in seen:
                seen.add(key)
                out.append(
                    topic_contract(
                        ctx,
                        name=dest,
                        broker=_BROKER,
                        kind=TOPIC_KIND_TOPIC,
                        role=role,
                        label=b_name,
                        origin=ContractOrigin.ROUTE_REGISTRATION,
                        offset=offset,
                    )
                )

        for ctx, target, offset in self._send_calls:
            dest = self._binding_destinations.get(target, target)
            key = (dest.lower(), "provider")
            if key not in seen:
                seen.add(key)
                out.append(
                    topic_contract(
                        ctx,
                        name=dest,
                        broker=_BROKER,
                        kind=TOPIC_KIND_TOPIC,
                        role="provider",
                        label="StreamBridge.send",
                        origin=ContractOrigin.ROUTE_REGISTRATION,
                        offset=offset,
                    )
                )

        return out


class SpringCloudStreamDialect:
    """Spring Cloud Stream bindings as topic contracts."""

    name = _BROKER
    extensions = _EXTENSIONS
    requires = _GATES

    def repo_pass(self) -> _CloudStreamPass:
        return _CloudStreamPass()

    def extract(self, ctx: ScanContext) -> list[Contract]:
        p = self.repo_pass()
        p.scan(ctx)
        return p.finish()


CLOUD_STREAM = SpringCloudStreamDialect()

__all__ = ["CLOUD_STREAM"]
