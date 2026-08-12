"""Turn the small LaTeX subset used in slide prose into typeset runs.

Slide body text carries short inline maths (``$\\rho$``, ``$A_{\\mathrm{od}}$``,
``$7\\times10^{4}$``) and ``**bold**`` emphasis.  Display equations go through
``equations.py`` instead; this module only has to handle what fits in a
sentence.

``parse(text)`` returns a list of ``Run`` — text plus the three attributes both
renderers understand: bold, italic, and baseline (normal / super / sub).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SYMBOLS = {
    r"\rho": "ρ", r"\theta": "θ", r"\Theta": "Θ", r"\beta": "β", r"\alpha": "α",
    r"\varphi": "φ", r"\phi": "φ", r"\pi": "π", r"\chi": "χ", r"\Delta": "Δ",
    r"\delta": "δ", r"\sigma": "σ", r"\tau": "τ", r"\lambda": "λ",
    r"\Psi": "Ψ", r"\psi": "ψ", r"\varepsilon": "ε", r"\epsilon": "ε",
    r"\mu": "μ", r"\nu": "ν", r"\Sigma": "Σ", r"\Omega": "Ω", r"\xi": "ξ",
    r"\leq": "≤", r"\geq": "≥", r"\ll": "≪", r"\gg": "≫", r"\neq": "≠",
    r"\times": "×", r"\approx": "≈", r"\otimes": "⊗", r"\cdot": "·",
    r"\pm": "±", r"\sim": "∼", r"\to": "→", r"\mapsto": "↦", r"\in": "∈",
    r"\langle": "⟨", r"\rangle": "⟩", r"\|": "‖", r"\infty": "∞",
    r"\ldots": "…", r"\dots": "…", r"\partial": "∂", r"\nabla": "∇",
    r"\dagger": "†", r"\circ": "∘", r"\propto": "∝", r"\equiv": "≡",
    r"\perp": "⊥", r"\cup": "∪", r"\cap": "∩", r"\subset": "⊂",
    r"\,": " ", r"\;": " ", r"\ ": " ", r"\!": "",
}
# multi-letter upright operators
WORDS = {r"\dim": "dim", r"\log": "log", r"\ln": "ln", r"\exp": "exp",
         r"\sin": "sin", r"\cos": "cos", r"\tanh": "tanh", r"\max": "max",
         r"\min": "min", r"\sum": "Σ", r"\Tr": "Tr", r"\mathrm": None,
         r"\mathcal": None, r"\text": None, r"\mathbf": None}

SUP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
       "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻", "+": "⁺", "(": "⁽", ")": "⁾"}


@dataclass
class Run:
    text: str
    bold: bool = False
    italic: bool = False
    baseline: str = ""          # "", "sup", "sub"


def _emit(out: list[Run], text: str, *, bold: bool, italic: bool, base: str) -> None:
    if not text:
        return
    if out and out[-1].bold == bold and out[-1].italic == italic \
            and out[-1].baseline == base:
        out[-1].text += text
    else:
        out.append(Run(text, bold, italic, base))


def _group(s: str, i: int) -> tuple[str, int]:
    """Read a braced group or a single token starting at i."""
    if i < len(s) and s[i] == "{":
        depth, j = 1, i + 1
        while j < len(s) and depth:
            depth += (s[j] == "{") - (s[j] == "}")
            j += 1
        return s[i + 1:j - 1], j
    if i < len(s) and s[i] == "\\":
        m = re.match(r"\\[A-Za-z]+", s[i:])
        if m:
            return s[i:i + m.end()], i + m.end()
    return (s[i], i + 1) if i < len(s) else ("", i)


def _math(s: str, out: list[Run], bold: bool, base: str = "",
          upright: bool = False) -> None:
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\[A-Za-z]+|\\.", s[i:])
            tok = m.group(0)
            if tok in (r"\mathrm", r"\text", r"\mathcal", r"\mathbf"):
                g, i = _group(s, i + len(tok))
                _math(g, out, bold or tok == r"\mathbf", base, upright=True)
                continue
            if tok in (r"\hat", r"\tilde", r"\bar", r"\vec", r"\dot"):
                g, i = _group(s, i + len(tok))     # accents have no plain-text form
                _math(g, out, bold, base, upright)
                continue
            if tok == r"\sqrt":
                g, i = _group(s, i + len(tok))
                _emit(out, "√", bold=bold, italic=False, base=base)
                _math(g, out, bold, base, upright)
                continue
            if tok in (r"\frac", r"\dfrac"):        # a/b, inline
                num, i = _group(s, i + len(tok))
                den, i = _group(s, i)
                _math(num, out, bold, base, upright)
                _emit(out, "/", bold=bold, italic=False, base=base)
                _math(den, out, bold, base, upright)
                continue
            if tok in WORDS and WORDS[tok]:
                _emit(out, WORDS[tok], bold=bold, italic=False, base=base)
            elif tok in SYMBOLS:
                _emit(out, SYMBOLS[tok], bold=bold, italic=False, base=base)
            else:                                   # unknown: strip the slash
                _emit(out, tok[1:], bold=bold, italic=not upright, base=base)
            i += len(tok)
            continue
        if c in "^_":
            g, i = _group(s, i + 1)
            _math(g, out, bold, "sup" if c == "^" else "sub", upright)
            continue
        if c == "{":
            g, i = _group(s, i)
            _math(g, out, bold, base, upright)
            continue
        if c == "}":
            i += 1
            continue
        if c.isalpha():
            _emit(out, c, bold=bold, italic=not upright, base=base)
        elif c == " ":
            _emit(out, " ", bold=bold, italic=False, base=base)
        else:
            _emit(out, c, bold=bold, italic=False, base=base)
        i += 1


def parse(text: str) -> list[Run]:
    """Split ``**bold**``, ``*italic*`` and ``$maths$`` into renderer-ready runs."""
    out: list[Run] = []
    bold = italic = False
    for chunk in re.split(r"(\*\*|\*)", text):
        if chunk == "**":
            bold = not bold
            continue
        if chunk == "*":
            italic = not italic
            continue
        for k, part in enumerate(re.split(r"\$", chunk)):
            if k % 2:
                _math(part, out, bold)
            else:
                _emit(out, part, bold=bold, italic=italic, base="")
    return [r for r in out if r.text]


def plain(text: str) -> str:
    """The same string with all markup removed — for alt text and logs."""
    return "".join(r.text for r in parse(text))


if __name__ == "__main__":
    for s in ["**The state is the input.** $\\rho$ enters directly",
              "$A_{\\mathrm{od}}$ and $7\\times10^{4}$ and $\\varphi_T$",
              "$\\mathrm{Tr}(\\rho S^2_{\\mathrm{od}})$, $10^{-6}$, $R_\\pm$",
              "$\\|\\rho_{\\mathrm{od}}\\|_F^2$, $\\beta^{-1}$, $\\pi$-systems"]:
        print(s, "->", "|".join(
            f"{r.text}{'/b' if r.bold else ''}{'/i' if r.italic else ''}"
            f"{'/' + r.baseline if r.baseline else ''}" for r in parse(s)))
