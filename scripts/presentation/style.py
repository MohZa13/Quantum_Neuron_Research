"""Shared visual language for the group-meeting deck.

One palette, one geometry, one set of matplotlib defaults, consumed by
``equations.py``, ``figures.py`` and ``build_deck.py`` so the slides and the
plots cannot drift apart.

The series colours are **not** hand-picked. They were snapped from the deck's
own brand hues onto steps that clear the six categorical-palette checks
(OKLCH lightness band, chroma floor, Machado-Oliveira-Fernandes 2009
protanopia/deuteranopia separation, normal-vision floor, contrast vs a white
surface). Slots 1-3 clear the *all-pairs* test and are therefore the cap for
scatter-like forms where any two marks can touch; slot 4 is legal only in
forms where adjacency is controlled (grouped/stacked bars, lines with direct
labels). Do not add a fifth slot: fold to "other", or facet.
"""
from __future__ import annotations

# --- ink and surfaces (text and chrome — never used to encode a series) ------
INK = "#1F2430"          # primary body text
NAVY = "#1E2761"         # headings, dark surfaces, display maths
NAVY_D = "#182050"       # dark-surface fill
SLATE = "#5A6478"        # secondary text, captions, axis labels
GRAY = "#9AA3B5"         # footers, de-emphasised chrome
PANEL = "#EEF3FB"        # pale panel fill
PANEL2 = "#E3EBF8"       # second-level panel fill
ONDARK = "#CADCFC"       # body text on a navy surface
HAIR = "#DDE3EE"         # gridlines, hairline rules
WHITE = "#FFFFFF"

# --- categorical series (fixed order, never cycled) -------------------------
SERIES = ["#3A5CA8", "#B85042", "#30A0C8", "#AD8524"]
BLUE, RUST, CYAN, GOLD = SERIES
SERIES_ALLPAIRS_CAP = 3

# a single-hue sequential ramp (magnitude), light -> dark, from the blue slot
SEQ = ["#D9E1F2", "#B3C4E4", "#8DA6D6", "#6788C7", "#3A5CA8", "#26407A"]

# --- typography -------------------------------------------------------------
# The pptx names the fonts the user's PowerPoint has; the HTML mirror falls
# back to metric-similar faces that exist in this sandbox.
SERIF = "Cambria"
SANS = "Calibri"
SERIF_STACK = "Cambria, 'Liberation Serif', 'Nimbus Roman', Georgia, serif"
SANS_STACK = "Calibri, 'Carlito', 'Liberation Sans', 'Noto Sans', sans-serif"

# --- slide geometry, inches (13.333 x 7.5 = 16:9) ---------------------------
SLIDE_W, SLIDE_H = 13.3333, 7.5
MX = 0.62                       # left/right margin
CW = SLIDE_W - 2 * MX           # content width
KICKER_Y, KICKER_SZ = 0.22, 9.0
TITLE_Y, TITLE_SZ = 0.46, 26.0
BODY_TOP = 1.28                 # first content row on a normal slide
FOOTER_Y, FOOTER_SZ = 7.02, 8.0
RADIUS = 0.06                   # panel corner radius, inches

# --- text sizes -------------------------------------------------------------
SZ_BODY = 13.0
SZ_SMALL = 11.0
SZ_CAPTION = 10.0
SZ_STAT = 27.0
SZ_CARD_TITLE = 13.5


def mpl_rc(dpi: int = 220) -> dict:
    """rcParams that make a matplotlib figure look like it belongs on a slide."""
    return {
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.family": "sans-serif",
        "font.sans-serif": ["Carlito", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "text.color": INK,
        "axes.edgecolor": HAIR,
        "axes.linewidth": 0.9,
        "axes.labelcolor": SLATE,
        "axes.titlecolor": NAVY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": HAIR,
        "grid.linewidth": 0.8,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "xtick.labelcolor": SLATE,
        "ytick.labelcolor": SLATE,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
    }
