"""
Digitize the classical FCIM (red dashed) curves from docs/fig_8.png.

The image is the paper's Fig. 8 (2x3 grid, n=2..7 qubits). We only need the
classical curve as a static reference -- the quantum curves are reproduced by
actually running the models (see run_pennylane_vs_original.py).

Calibration (pixel <-> data) was derived by detecting the axes bounding boxes
and tick-mark pixel positions programmatically, then reading the tick labels
off zoomed crops of the image. Recorded here as fixed constants rather than
re-derived every run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_PATH = REPO_ROOT / "docs" / "fig_8.png"
OUT_DIR = REPO_ROOT / "results" / "digitized"

# panel pixel boxes: (x0, x1, y0, y1) = axes-spine bounding box
PANEL_BOXES = {
    2: (100, 450, 53, 232),
    3: (564, 914, 53, 232),
    4: (1027, 1377, 53, 232),
    5: (100, 450, 341, 521),
    6: (564, 914, 341, 521),
    7: (1027, 1377, 341, 521),
}

# x-axis tick pixel centers (all panels share the same 0..500 epoch axis)
PANEL_XTICKS_PX = {
    2: [116, 180, 244, 307, 371, 435],
    3: [580, 643, 707, 771, 835, 898],
    4: [1043, 1107, 1171, 1234, 1298, 1362],
    5: [116, 180, 244, 307, 371, 435],
    6: [580, 643, 707, 771, 835, 898],
    7: [1043, 1107, 1171, 1234, 1298, 1362],
}
XTICK_VALUES = [0, 100, 200, 300, 400, 500]

# y-axis tick pixel centers + values, read off zoomed crops of each panel
PANEL_YTICKS_PX = {
    2: [79, 106, 132, 159, 185, 212],
    3: [56, 77, 98, 119, 141, 162, 183, 204, 225],
    4: [62, 84, 106, 128, 149, 171, 193, 215],
    5: [345, 366, 387, 408, 429, 451, 472, 493, 514],
    6: [353, 375, 396, 418, 439, 461, 482, 504],
    7: [358, 381, 404, 428, 451, 474, 498],
}
PANEL_YTICK_VALUES = {
    2: [1.40, 1.38, 1.36, 1.34, 1.32, 1.30],
    3: [1.48, 1.45, 1.43, 1.40, 1.38, 1.35, 1.32, 1.30, 1.27],
    4: [1.43, 1.40, 1.38, 1.35, 1.33, 1.30, 1.28, 1.25],
    5: [1.60, 1.55, 1.50, 1.45, 1.40, 1.35, 1.30, 1.25, 1.20],
    6: [1.55, 1.50, 1.45, 1.40, 1.35, 1.30, 1.25, 1.20],
    7: [1.50, 1.45, 1.40, 1.35, 1.30, 1.25, 1.20],
}


def _linear_calibration(px, val):
    """Least-squares fit of pixel -> data value (both are linear axes)."""
    px = np.asarray(px, dtype=float)
    val = np.asarray(val, dtype=float)
    slope, intercept = np.polyfit(px, val, 1)
    return slope, intercept


def digitize_panel(n: int, arr: np.ndarray) -> pd.DataFrame:
    x0, x1, y0, y1 = PANEL_BOXES[n]
    x_slope, x_intercept = _linear_calibration(PANEL_XTICKS_PX[n], XTICK_VALUES)
    y_slope, y_intercept = _linear_calibration(PANEL_YTICKS_PX[n], PANEL_YTICK_VALUES[n])

    # red dashed classical curve: strong red channel, weak green/blue
    region = arr[y0 + 1 : y1, x0 + 1 : x1]
    r = region[:, :, 0].astype(int)
    g = region[:, :, 1].astype(int)
    b = region[:, :, 2].astype(int)
    red_mask = (r > 150) & (g < 140) & (b < 140) & (r - g > 60) & (r - b > 60)

    # blank out the legend box (contains a red-dashed sample swatch that would
    # otherwise be mistaken for the curve); box is at a fixed relative offset
    # in every panel since the legend placement/size is identical throughout.
    red_mask[0:55, 125:350] = False

    n_cols = region.shape[1]
    col_px_x = np.arange(x0 + 1, x1)
    epoch_at_col = x_slope * col_px_x + x_intercept

    rows_per_col = []
    valid_cols = []
    for c in range(n_cols):
        rows = np.where(red_mask[:, c])[0]
        if rows.size:
            rows_per_col.append(rows.mean() + y0 + 1)
            valid_cols.append(c)

    valid_epochs = epoch_at_col[valid_cols]
    valid_rows_px = np.array(rows_per_col)

    # dashed line leaves gaps -- interpolate over the full 0..500 epoch grid
    order = np.argsort(valid_epochs)
    valid_epochs = valid_epochs[order]
    valid_rows_px = valid_rows_px[order]

    epoch_grid = np.arange(0, 501)
    row_px_interp = np.interp(epoch_grid, valid_epochs, valid_rows_px)
    classical_loss = y_slope * row_px_interp + y_intercept

    return pd.DataFrame({"Epoch": epoch_grid, "Classical_Loss_Digitized": classical_loss})


def main() -> None:
    im = Image.open(IMG_PATH).convert("RGB")
    arr = np.array(im)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in range(2, 8):
        df = digitize_panel(n, arr)
        out_path = OUT_DIR / f"fig8_classical_{n}qubit_digitized.csv"
        df.to_csv(out_path, index=False)
        print(f"n={n}: {out_path}  "
              f"(start={df['Classical_Loss_Digitized'].iloc[0]:.4f}, "
              f"end={df['Classical_Loss_Digitized'].iloc[-1]:.4f})")


if __name__ == "__main__":
    main()
