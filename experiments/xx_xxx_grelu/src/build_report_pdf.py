"""Render REPORT.md to REPORT.pdf (Markdown -> HTML -> headless Chrome).

    python3 experiments/xx_xxx_grelu/src/build_report_pdf.py

Needs the `markdown` package and Google Chrome. Figures are embedded from
figures/ by their relative paths in the Markdown.
"""

import pathlib
import subprocess
import tempfile

import markdown

EXP = pathlib.Path(__file__).resolve().parent.parent
SRC, OUT = EXP / "REPORT.md", EXP / "REPORT.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
html { font-size: 10pt; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color: #111;
       line-height: 1.45; max-width: none; margin: 0; }
h1 { font-size: 19pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; padding-bottom: 3pt;
     border-bottom: 1px solid #ccc; break-after: avoid; }
h3 { font-size: 11pt; margin: 13pt 0 4pt; break-after: avoid; }
p, li { margin: 4pt 0; }
ul, ol { padding-left: 16pt; }
em { color: #444; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.6pt;
       background: #f3f3f1; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f6f6f4; border: 1px solid #e2e2de; border-radius: 3pt;
      padding: 6pt 8pt; font-size: 8.4pt; line-height: 1.35; overflow: hidden;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; font-size: 8.8pt;
        break-inside: avoid; }
th, td { border: 1px solid #d6d6d2; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f1f1ee; }
img { display: block; max-width: 100%; margin: 6pt auto 4pt; break-inside: avoid; }
p:has(> img) { break-inside: avoid; }
p:has(+ ul), p:has(+ ol) { break-after: avoid; }
li:has(> ul), li:has(> ol) { break-inside: avoid; }
hr { border: none; border-top: 1px solid #ccc; margin: 12pt 0; }
"""


def main():
    md = SRC.read_text()
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>GReLU neuron report</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    # the HTML must sit in EXP so figures/… resolves
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=EXP, delete=False) as f:
        f.write(html)
        tmp = pathlib.Path(f.name)
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--allow-file-access-from-files", f"--print-to-pdf={OUT}", tmp.as_uri()],
                       check=True, capture_output=True, timeout=120)
    finally:
        tmp.unlink()
    print(f"wrote {OUT.relative_to(EXP.parent.parent)}")


if __name__ == "__main__":
    main()
