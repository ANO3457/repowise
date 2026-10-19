"""Owl mascot: banner art, eye states, and the block wordmark.

Single home for every piece of the init-banner identity: the owl, the REPOWISE
block wordmark, the mini owl and its states, and the spinner that blinks while
repowise thinks.

All render functions are pure and deterministic, and there is exactly one size
of everything. The banner used to ship two — a full variant at 2-char strokes
and a compact one at 1 — with the caller hardcoding compact and the width maths
measuring full, so on an 80-column terminal the tagline shortened to fit art
that was 47 columns wide. Nothing rendered the full art in its whole life, so it
is gone rather than fixed: one size cannot desync from itself.

The wordmark is painted in the one brand colour. It used to run a five-shade
"heatmap" ramp that swept left to right and then jittered each cell by ±1.1
shades. Across the 31-cell wordmark the sweep moves 0.133 shades per cell while
the jitter moves up to 2.2, so the gradient never survived: counted over the
real grid, all 155 painted cells landed near-uniformly across the five shades.
It rendered as orange static in the first thing a user ever sees, which is the
"decorate only what does something" rule with nothing behind the paint.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.text import Text

from repowise.cli.ui.brand import BRAND_STYLE
from repowise.cli.ui.brand import OWL as OWL_STYLE
from repowise.cli.ui.glyphs import ascii_only

# ---------------------------------------------------------------------------
# Owl
# ---------------------------------------------------------------------------

OWL = [
    " ,___,",
    " (◉,◉)",
    " ( ▼ )",
    " /)_(\\",
    '  " " ',
]

# The mini owl is the big owl's face line. Its eyes say how the run went, so a
# state reads with colour off. state -> (unicode eyes, ascii eyes).
IDLE = "idle"
BLINK = "blink"
SHUT = "shut"
DONE = "done"
INTERRUPTED = "interrupted"
WARNED = "warn"
FAILED = "failed"

EYES: dict[str, tuple[str, str]] = {
    IDLE: ("◉", "o"),
    BLINK: ("◒", "o"),
    SHUT: ("─", "o"),
    DONE: ("^", "^"),
    INTERRUPTED: ("─", "-"),
    WARNED: ("•", "o"),
    FAILED: ("×", "x"),
}

# ASCII forms of the banner art. One character each, so every span stays put.
_ASCII_ART = str.maketrans({"◉": "o", "▼": "v", "█": "#", "▄": "#", "▀": "#"})


def _eyes(state: str) -> str:
    unicode, ascii_form = EYES[state]
    return ascii_form if ascii_only() else unicode


def mini(state: str = IDLE) -> str:
    """The mini owl face for *state*, e.g. ``(◉,◉)``. Always 5 columns."""
    eyes = _eyes(state)
    return f"({eyes},{eyes})"


def mini_text(state: str = IDLE) -> Text:
    """:func:`mini` with cream strokes and orange eyes."""
    eyes = _eyes(state)
    return Text.assemble(
        ("(", OWL_STYLE),
        (eyes, BRAND_STYLE),
        (",", OWL_STYLE),
        (eyes, BRAND_STYLE),
        (")", OWL_STYLE),
    )


# The spinner is a slow blink, not a spin. A continuous eye-roll at 5fps reads
# as a cheap loading GIF and draws the eye away from the progress bar next to
# it; holding the eyes open and closing them once per cycle reads as alive
# without competing for attention. Motion lives in the last three frames.
_BLINK_CYCLE = [IDLE] * 7 + [BLINK, SHUT, BLINK]
_FRAME_SECONDS = 0.18

THINKING_FRAMES = [mini(state) for state in _BLINK_CYCLE]


def thinking_text(time: float) -> Text:
    """The spinner frame for *time*, in seconds."""
    return mini_text(_BLINK_CYCLE[int(time / _FRAME_SECONDS) % len(_BLINK_CYCLE)])


class OwlSpinner:
    """The blinking mini owl in front of a status line; see :func:`owl_status`."""

    def __init__(self, text: str = "") -> None:
        self.text = Text.from_markup(text)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        frame = thinking_text(console.get_time())
        yield Text.assemble(frame, " ", self.text) if self.text else frame


def owl_status(console: Console, text: str, **live_kwargs: Any) -> Live:
    """``console.status`` with the owl: a transient line that blinks while work runs.

    Built the way Rich builds a status, so it behaves the same on a pipe.
    """
    return Live(
        OwlSpinner(text),
        console=console,
        refresh_per_second=12.5,
        transient=True,
        **live_kwargs,
    )


# ---------------------------------------------------------------------------
# Wordmark font — half-block letterforms at 1-char strokes (R tail = stepped
# taper).
# ---------------------------------------------------------------------------

FONT = {
    "R": ["███▄ ", "█  █ ", "███▀ ", "█ ▀▄ ", "█  ▀▄"],
    "E": ["███", "█  ", "██ ", "█  ", "███"],
    "P": ["███▄", "█  █", "███▀", "█   ", "█   "],
    "O": ["▄██▄", "█  █", "█  █", "█  █", "▀██▀"],
    "W": ["█   █", "█   █", "█ █ █", "█ █ █", "▀█▀█▀"],
    "I": ["███", " █ ", " █ ", " █ ", "███"],
    "S": ["▄██▄", "█   ", "▀██▄", "   █", "▀██▀"],
}

_WORD = "REPOWISE"
_LETTER_GAP = 1
_OWL_GAP = 2
_INDENT = 1


@cache
def render_wordmark() -> tuple[str, ...]:
    """The five wordmark rows, letters separated by a single column."""
    rows = ["", "", "", "", ""]
    for index, char in enumerate(_WORD):
        glyph = FONT[char]
        for r in range(5):
            rows[r] += glyph[r]
            if index < len(_WORD) - 1:
                rows[r] += " " * _LETTER_GAP
    return tuple(rows)


def banner_width() -> int:
    """Total rendered banner width in columns (indent + owl + gap + wordmark).

    Single source of truth for layout maths, and now actually single: the
    tagline switch in ``print_banner`` calls this, and there is no second
    variant left for it to measure by mistake.
    """
    return _INDENT + max(len(line) for line in OWL) + _OWL_GAP + len(render_wordmark()[0])


def _paint_owl(line: str) -> Text:
    """Owl strokes in cream, eyes in brand orange."""
    out = Text()
    for ch in line:
        if ch == EYES[IDLE][0]:
            out.append(ch, style=BRAND_STYLE)
        elif ch == " ":
            out.append(" ")
        else:
            out.append(ch, style=OWL_STYLE)
    return out


def banner_text() -> Text:
    """Compose owl + wordmark into a single renderable ``Text``."""
    rows = render_wordmark()
    owl_w = max(len(line) for line in OWL)
    out = Text()
    for r in range(5):
        if r:
            out.append("\n")
        out.append(" " * _INDENT)
        out.append_text(_paint_owl(OWL[r].ljust(owl_w)))
        out.append(" " * _OWL_GAP)
        out.append(rows[r], style=BRAND_STYLE)
    if ascii_only():
        out.plain = out.plain.translate(_ASCII_ART)
    return out
