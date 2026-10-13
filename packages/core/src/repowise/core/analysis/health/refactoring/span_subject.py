"""What an Extract Method span is about, read off its code, for the helper's name.

The last naming source, after a stage label, a banner and ``compute_<out>``
(:mod:`.helper_naming`): the host function's name qualified by the span's
subject, ``render_card`` + ``directive`` -> ``render_card_directive``. Both
halves are facts the plan holds, so the name says where the helper came from
and what it handles without guessing a verb for what it does.

The subject, first found: the span's single OUT value; the condition of the
``if`` arm or the target of the loop whose whole body the span is; then for
each of the span's statements in order, an ``if`` condition's subject (a string
key or compared literal, else the last name read), a loop's iterable, the
first name a statement binds, or a dropped call's name. Role words (``result``,
``self``) and one-letter names say nothing and are skipped.

What the span does (:func:`actions`) is read first: the call a top-level
statement makes, when its name opens with a verb. Its own words name it
(``load_settings``); a one-word call adds what it binds or reads
(``verify_password``). The span's single product picks the call.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..dataflow.effects import binding_target, bound_names
from ..dataflow.span import body_holder, is_comment
from .naming import split_words

if TYPE_CHECKING:
    from ..complexity.languages import LanguageNodeMap

# Names that describe a role or a receiver, not what a block handles.
_ROLE_WORDS = frozenset(
    {"out", "result", "results", "value", "values", "ret", "tmp", "temp", "data", "item"}
    | {"self", "this", "cls", "none", "null", "undefined", "true", "false", "args", "kwargs"}
    # ``typeof x === "number"`` compares a type name, not the subject.
    | {"number", "string", "boolean", "object", "function", "bigint", "symbol"}
)
# A dropped call to one of these adds to something; the call's name says
# nothing about what (``out.append(...)``), so the next statement is read.
_GENERIC_CALLS = frozenset(
    {"append", "push", "add", "extend", "insert", "update", "set", "write", "print", "echo"}
    | {"log", "info", "debug", "warning", "warn", "error", "emit"}
)
# Calls that pass their subject through: ``card.get("owner")`` is about
# ``owner``, ``enumerate(groups)`` about ``groups``. ``_GENERIC_CALLS`` do too.
_PASS_THROUGH = frozenset(
    {"get", "items", "keys", "values", "enumerate", "sorted", "reversed", "list", "tuple"}
    | {"len", "iter", "getattr", "isinstance", "bool", "str", "int", "object", "array"}
)
_MEMBER_FIELDS = ("attribute", "property", "field")
_CALLEE_FIELDS = ("function", "callee")
_LOOP_SOURCE_FIELDS = ("right", "value", "iterable", "condition")
_VALUE_FIELDS = ("right", "value")
_PLAIN_NAME = re.compile(r"[A-Za-z_$][\w$]*")
_MAX_DEPTH = 8
# A fixed English list, never read from the repository: words that head a
# function name as a verb far more often than as a noun (``list``, ``match``,
# ``record`` head nouns as often, so they are out).
_VERBS = frozenset(
    {"get", "set", "read", "write", "load", "save", "parse", "format", "build", "make"}
    | {"create", "init", "compute", "calculate", "calc", "resolve", "find", "fetch", "update"}
    | {"add", "remove", "delete", "clear", "reset", "apply", "check", "validate", "verify"}
    | {"ensure", "normalize", "convert", "collect", "gather", "filter", "sort", "merge", "split"}
    | {"join", "render", "send", "open", "close", "start", "stop", "run", "execute", "handle"}
    | {"register", "install", "import", "export", "extract", "insert", "upsert", "prepare"}
    | {"configure", "attach", "detach", "bind", "encode", "decode", "serialize", "deserialize"}
    | {"compile", "generate", "transform", "expand", "scan", "search", "select", "pick", "wrap"}
    | {"unwrap", "strip", "sanitize", "escape", "copy", "clone", "move", "rename", "replace"}
    | {"refresh", "sync", "flush", "dispatch", "notify", "publish", "subscribe", "poll", "wait"}
    | {"retry", "track", "measure", "estimate", "compare", "emit", "spawn", "launch", "cancel"}
    | {"abort", "enable", "disable", "toggle", "describe", "summarize", "show", "hide", "draw"}
    | {"mount", "connect", "disconnect", "authenticate", "authorize", "download", "upload"}
    | {"persist", "restore", "infer", "detect", "identify", "classify", "partition", "walk"}
    | {"visit", "traverse", "fill", "derive", "assign", "allocate", "acquire", "unlock", "drain"}
    | {"decorate", "inject", "evaluate", "invoke", "inspect", "prune", "trim", "truncate"}
    | {"harden", "encrypt", "decrypt", "compress", "decompress", "migrate", "upgrade", "rebuild"}
    | {"clean", "warn", "rewrite", "discover", "consume", "produce", "reload", "redact"}
)
# Verbs that promise a value; ``calc`` is in, as short for ``calculate``.
VALUE_VERBS = frozenset({"compute", "calculate", "calc", "derive", "get", "build", "make"})
# The receiver a method is called on when it is the host's own object: the
# helper would sit beside, or override, the method it calls.
_SELF_RECEIVERS = frozenset({"self", "this", "cls"})
# A bound name that is a flag or a raw handle says how, not what:
# ``is_correct = hasher.verify(password)`` is about ``password``.
_FLAG_WORDS = frozenset({"is", "has", "should", "can", "did", "was", "raw", "res", "resp", "ok"})
_MAX_NOUN_WORDS = 3
# What a call's value is read through: ``await f()``, ``(f())``, ``void f()``.
_WRAPPER_KINDS = frozenset(
    {"await_expression", "parenthesized_expression", "unary_expression", "non_null_expression"}
    | {"as_expression", "satisfies_expression"}
)
# String methods: ``name.strip()`` is about ``name``, not stripping.
_STRING_OPS = frozenset(
    {"strip", "lstrip", "rstrip", "trim", "trim_start", "trim_end", "lower", "upper", "casefold"}
    | {"to_lower_case", "to_upper_case", "splitlines", "startswith", "endswith", "to_string"}
)
# Arguments that hold no subject of their own.
_NOT_A_SUBJECT = frozenset(
    {"keyword_argument", "object", "dictionary", "lambda", "arrow_function", "function_expression"}
    | {"list_splat", "dictionary_splat", "spread_element"}
)


def host_words(name: str) -> list[str]:
    """The host function's name as words; ``[]`` for an anonymous or dotted one."""
    return split_words(name) if _PLAIN_NAME.fullmatch(name or "") else []


def subjects(stmts: list[Any], returns: tuple[str, ...], lmap: LanguageNodeMap) -> list[list[str]]:
    """The span's subjects, best first, each as words, without repeats."""
    found: list[list[str]] = []
    if len(returns) == 1:
        found.append(_words(returns[0]))
    found.append(_enclosing_subject(stmts, lmap))
    code = [s for s in stmts if not is_comment(s)]
    for stmt in code:
        found.append(_statement_subject(stmt, lmap))
        if stmt.type in lmap.loop_kinds:
            # Over the host's own subject, the element is what the span handles.
            found.append(_expr_words(stmt.child_by_field_name("left"), 0))
    out: list[list[str]] = []
    for words in found:
        if words and words not in out:
            out.append(words)
    return out


def _words(text: str) -> list[str]:
    words = split_words(text)
    if not words or "_".join(words) in _ROLE_WORDS or all(len(w) == 1 for w in words):
        return []
    return words


def _enclosing_subject(stmts: list[Any], lmap: LanguageNodeMap) -> list[str]:
    """The subject of the ``if`` arm or loop whose whole body *stmts* is."""
    holder = body_holder(stmts)
    if holder is None:
        return []
    if holder.type in lmap.loop_kinds:
        return _expr_words(holder.child_by_field_name("left"), 0)
    return _expr_words(holder.child_by_field_name("condition"), 0)


def _statement_subject(stmt: Any, lmap: LanguageNodeMap) -> list[str]:
    kind = stmt.type
    if kind in lmap.if_kinds:
        return _expr_words(stmt.child_by_field_name("condition"), 0)
    if kind in lmap.loop_kinds:
        return _expr_words(_child(stmt, _LOOP_SOURCE_FIELDS), 0)
    if kind in lmap.try_kinds or kind in lmap.with_kinds:
        return _wrapper_subject(stmt, lmap)
    expr = stmt.named_children[0] if stmt.named_child_count == 1 else stmt
    if expr.type in lmap.await_kinds | {"await_expression"} and expr.named_child_count == 1:
        expr = expr.named_children[0]
    bound = _bound_subject(expr, lmap)
    if bound is not None:
        return bound
    if expr.type not in lmap.call_kinds or _generic(_callee_words(expr, 0)):
        # ``out.append(...)`` says nothing about what is added: read on.
        return []
    return _expr_words(expr, 0)


def _wrapper_subject(stmt: Any, lmap: LanguageNodeMap) -> list[str]:
    """A ``try`` or ``with`` is about what its body opens with."""
    body = stmt.child_by_field_name("body") or next(
        (c for c in stmt.named_children if c.type in lmap.block_kinds), None
    )
    first = next((c for c in body.named_children if not is_comment(c)), None) if body else None
    return _statement_subject(first, lmap) if first is not None else []


def _bound_subject(expr: Any, lmap: LanguageNodeMap) -> list[str] | None:
    """``x = ...`` is about ``x``, ``state.registry = ...`` about ``registry``;
    ``None`` when *expr* binds nothing."""
    holders = [expr, *expr.named_children]
    target = next((t for h in holders if (t := binding_target(h, lmap)) is not None), None)
    if target is None:
        return None
    names = bound_names(target)
    return _words(names[0]) if names else _expr_words(target, 0)


def _expr_words(node: Any, depth: int) -> list[str]:
    """The name an expression is about (module docstring)."""
    if node is None or depth > _MAX_DEPTH:
        return []
    if "string" in node.type:
        return _literal_words(node)
    if not node.named_children:
        text = (node.text or b"").decode("utf-8", "replace")
        return _words(text) if _PLAIN_NAME.fullmatch(text) else []
    if node.type in ("call", "call_expression"):
        return _call_words(node, depth)
    member = _child(node, _MEMBER_FIELDS)
    # A member names its field; otherwise a compared key, then the first operand.
    keys = [c for c in node.named_children if "string" in c.type]
    parts = [member] if member is not None else [*keys, *node.named_children]
    return next((w for c in parts if (w := _expr_words(c, depth + 1))), [])


def _callee_words(call: Any, depth: int) -> list[str]:
    return _expr_words(_child(call, _CALLEE_FIELDS), depth + 1)


def _generic(words: list[str]) -> bool:
    return "_".join(words) in _GENERIC_CALLS


def _call_words(call: Any, depth: int) -> list[str]:
    """A call is about its name, or, when the name only passes a value on
    (``get``, ``len``, ``append``), its first argument, then its receiver."""
    name = _callee_words(call, depth)
    if name and not _generic(name) and "_".join(name) not in _PASS_THROUGH:
        return name
    first, receiver = _argument(call), _called(call)[1]
    return next((w for c in (first, receiver) if (w := _expr_words(c, depth + 1))), [])


def _child(node: Any, fields: tuple[str, ...]) -> Any:
    return next((c for f in fields if (c := node.child_by_field_name(f)) is not None), None)


def _literal_words(literal: Any) -> list[str]:
    """A string literal that is a plain key (``"recent_owner"``), as words."""
    text = (literal.text or b"").decode("utf-8", "replace").strip("'\"`")
    return _words(text) if _PLAIN_NAME.fullmatch(text) else []


@dataclass(frozen=True)
class _KeyCall:
    """A statement's call as naming reads it. ``free``: no receiver, or the
    host's own, so its exact name would collide. ``bound``: the one name the
    result is bound to; ``target`` and ``noun`` are the words a one-word call
    takes."""

    callee: list[str]
    free: bool
    dropped: bool
    args: bool
    bound: str | None
    target: list[str]
    noun: list[str]


def actions(stmts: list[Any], returns: tuple[str, ...], lmap: LanguageNodeMap) -> list[list[str]]:
    """What the span does, as verb-first words, best first (module docstring)."""
    code = list(_top_statements(stmts, lmap, 0))
    calls = [c for s in code if (c := _key_call(s, lmap)) is not None]
    product = _product(code, returns, lmap)
    if product:
        # One product: the last call binding it; the others serve it.
        ordered = [c for c in reversed(calls) if c.bound == product][:1]
    else:
        # A call run for what it does, first; else what the span binds last.
        ordered = [c for c in calls if c.dropped] + [c for c in reversed(calls) if not c.dropped]
        # ``session.commit()`` closes a step; it does not say what the step is.
        ordered.sort(key=lambda c: len(c.callee) == 1 and not c.args)
    out: list[list[str]] = []
    for call in ordered:
        words = _action_words(call)
        if words and words not in out:
            out.append(words)
    return out


def _product(code: list[Any], returns: tuple[str, ...], lmap: LanguageNodeMap) -> str | None:
    """The one value the span makes: its single OUT value, else the name its
    last statement returns; ``None`` for a role word or without one."""
    name = returns[0] if len(returns) == 1 else None
    last = code[-1] if code else None
    if name is None and last is not None and last.type in lmap.return_kinds and last.named_children:
        name = _node_text(last.named_children[0])
    return name if name and _PLAIN_NAME.fullmatch(name) and _words(name) else None


def _top_statements(stmts: list[Any], lmap: LanguageNodeMap, depth: int) -> Iterator[Any]:
    """*stmts*' code statements, a ``try`` or ``with`` read through to its body."""
    for stmt in stmts:
        if is_comment(stmt):
            continue
        if depth < 2 and (stmt.type in lmap.try_kinds or stmt.type in lmap.with_kinds):
            body = _body(stmt, lmap)
            if body is not None:
                yield from _top_statements(body.named_children, lmap, depth + 1)
            continue
        yield stmt


def _body(stmt: Any, lmap: LanguageNodeMap) -> Any:
    return stmt.child_by_field_name("body") or next(
        (c for c in stmt.named_children if c.type in lmap.block_kinds), None
    )


def _key_call(stmt: Any, lmap: LanguageNodeMap) -> _KeyCall | None:
    """The call *stmt* makes for its result or drops, else ``None``: a
    constructor (``Resolver(...)``) or a call that only passes a value on
    (``list(...)``, ``out.append(...)``, ``name.strip()``) is read through or
    skipped."""
    expr = stmt.named_children[0] if stmt.named_child_count == 1 else stmt
    holders = (expr, *expr.named_children)
    binder = next((h for h in holders if binding_target(h, lmap) is not None), None)
    value = _unwrap(_child(binder, _VALUE_FIELDS) if binder is not None else expr, lmap)
    if value is None:
        return None
    name, receiver = _called(value)
    if not _PLAIN_NAME.fullmatch(name) or name.lstrip("_$")[:1].isupper():
        return None
    target = binding_target(binder, lmap) if binder is not None else None
    names = bound_names(target) if target is not None else []
    return _KeyCall(
        callee=split_words(name),
        free=receiver is None or _node_text(receiver) in _SELF_RECEIVERS,
        dropped=binder is None and stmt.type not in lmap.return_kinds,
        args=_first_argument(value) is not None,
        bound=names[0] if len(names) == 1 else None,
        target=_target_words(target, names),
        noun=_noun(value, receiver),
    )


def _target_words(target: Any, names: list[str]) -> list[str]:
    """What a binding binds: one name, or a field (``self.stats``); ``[]`` for
    a destructuring (``a, b = f()``), which names no one thing."""
    if len(names) == 1:
        return _words(names[0])
    return _expr_words(target, 0) if target is not None and not names else []


def _noun(call: Any, receiver: Any) -> list[str]:
    """The subject of *call*'s first argument, else of its receiver."""
    parts = (_first_argument(call), receiver)
    return next((w for c in parts if (w := _expr_words(_plain(c), 0))), [])


def _called(call: Any) -> tuple[str, Any]:
    """The name *call* calls and its receiver (``None`` for a free function)."""
    callee = _child(call, _CALLEE_FIELDS)
    member = _child(callee, _MEMBER_FIELDS) if callee is not None else None
    if member is None:
        return _node_text(callee), None
    return _node_text(member), _child(callee, ("object", "value"))


def _plain(node: Any) -> Any:
    """*node* without the string methods called on it: ``text.strip()`` is ``text``."""
    while node is not None and node.type in ("call", "call_expression"):
        called, receiver = _called(node)
        if receiver is None or "_".join(split_words(called)) not in _STRING_OPS:
            break
        node = receiver
    return node


def _unwrap(node: Any, lmap: LanguageNodeMap) -> Any:
    """The call *node* evaluates, through ``await``, parentheses and calls that
    pass a value on; ``None`` when it is not a call or is a string's method."""
    for _ in range(_MAX_DEPTH):
        if node is None:
            return None
        if node.type in _WRAPPER_KINDS | lmap.await_kinds and node.named_children:
            node = node.named_children[0]
            continue
        if node.type not in lmap.call_kinds:
            return None
        called, receiver = _called(node)
        name = "_".join(split_words(called))
        if receiver is not None and "string" in receiver.type:
            return None
        if name in _STRING_OPS:
            node = receiver
        elif _generic([name]) or name in _PASS_THROUGH:
            node = _first_argument(node)
        else:
            return node
    return None


def _argument(call: Any) -> Any:
    """*call*'s first argument, whatever it is."""
    args = call.child_by_field_name("arguments")
    return next((a for a in args.named_children if not is_comment(a)), None) if args else None


def _first_argument(call: Any) -> Any:
    """The first positional argument, ``None`` for a keyword, object or callback."""
    first = _argument(call)
    return None if first is None or first.type in _NOT_A_SUBJECT else first


def _action_words(call: _KeyCall) -> list[str]:
    """Verb then noun for *call*, ``[]`` when its name opens with no verb.

    A call's own words name it; a free one drops its first qualifier so the
    helper does not take the name it calls. A one-word call adds the name it
    binds, else its argument or receiver."""
    verb, rest = call.callee[0], call.callee[1:]
    if verb not in _VERBS:
        return []
    if rest:
        noun = rest[1:] if call.free else rest
        return [verb, *noun[-_MAX_NOUN_WORDS:]] if noun else []
    target = [] if call.target[:1] and call.target[0] in _FLAG_WORDS else call.target
    noun = _drop_stem(verb, target) or call.noun
    return [verb, *noun[-_MAX_NOUN_WORDS:]] if noun else []


def action_verbs(stmts: list[Any], lmap: LanguageNodeMap) -> frozenset[str]:
    """The verbs the span's key calls open with."""
    calls = (_key_call(s, lmap) for s in _top_statements(stmts, lmap, 0))
    return frozenset(c.callee[0] for c in calls if c is not None and c.callee[0] in _VERBS)


def _drop_stem(verb: str, words: list[str]) -> list[str]:
    """``normalize`` + ``normalized_tools`` is ``normalize_tools``."""
    return words[1:] if words[:1] and words[0].startswith(verb) else words


def _node_text(node: Any) -> str:
    return (node.text or b"").decode("utf-8", "replace") if node is not None else ""


__all__ = ["VALUE_VERBS", "action_verbs", "actions", "host_words", "subjects"]
