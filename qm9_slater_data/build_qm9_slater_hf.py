"""Compatibility wrapper for the renamed QH9 Slater builder.

Use build_qh9_slater.py for new runs.  This file remains only so older
commands that referenced the QM9/HF-era filename still dispatch to the current
QH9 Hamiltonian workflow.
"""

import warnings
from pathlib import Path
import runpy
import sys

_QH9_SCRIPT = Path(__file__).resolve().parents[1] / "qh9_slater_data" / "build_qh9_slater.py"
_QH9_GLOBALS = runpy.run_path(str(_QH9_SCRIPT))

globals().update(
    {
        name: value
        for name, value in _QH9_GLOBALS.items()
        if not name.startswith("__") and name not in {"main"}
    }
)

_qh9_main = _QH9_GLOBALS["main"]


def main():
    if not any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        warnings.warn(
            "build_qm9_slater_hf.py has been renamed to build_qh9_slater.py; "
            "running the QH9 Hamiltonian workflow.",
            DeprecationWarning,
            stacklevel=2,
        )
    _qh9_main()


if __name__ == "__main__":
    main()
