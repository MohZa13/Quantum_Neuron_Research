"""Create paper-style figures from the 2/4/7 comparison benchmark CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "tests"))
from notebook_test_utils import REPO_ROOT


BENCHMARK_DIRECTORY = REPO_ROOT / "benchmarks"
PLOT_DIRECTORY = REPO_ROOT / "figures"

METHOD_LABELS = {
    "original_dense": "Original dense",
    "original_dense_probe": "Original dense probe",
    "penny_exact_aggregate": "PennyLane exact aggregate",
    "penny_chebyshev_matrix_free": "PennyLane Chebyshev",
    "penny_diagonal_fcim": "PennyLane diagonal FCIM",
}

METHOD_COLORS = {
    "original_dense": "#111111",
    "original_dense_probe": "#555555",
    "penny_exact_aggregate": "#0057B8",
    "penny_chebyshev_matrix_free": "#2E8B57",
    "penny_diagonal_fcim": "#D62728",
}

METHOD_STYLES = {
    "original_dense": "-",
    "original_dense_probe": "--",
    "penny_exact_aggregate": "-",
    "penny_chebyshev_matrix_free": "-.",
    "penny_diagonal_fcim": "--",
}


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_data(suffix: str):
    curves_path = BENCHMARK_DIRECTORY / f"paper_training_curves_{suffix}.csv"
    summary_path = BENCHMARK_DIRECTORY / f"paper_efficiency_summary_{suffix}.csv"
    sampling_path = BENCHMARK_DIRECTORY / f"paper_sampling_efficiency_{suffix}.csv"
    if not curves_path.exists():
        raise FileNotFoundError(
            f"Missing {curves_path}. Run benchmark_paper_comparison.py first."
        )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run benchmark_paper_comparison.py first."
        )
    curves = pd.read_csv(curves_path)
    summary = pd.read_csv(summary_path)
    sampling = pd.read_csv(sampling_path) if sampling_path.exists() else None
    return curves, summary, sampling


def finalize_axis(ax, *, title: str, ylabel: str | None = None):
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    if ylabel:
        ax.set_ylabel(ylabel)


def plot_training_curves(curves: pd.DataFrame, suffix: str) -> Path:
    qubits = sorted(curves["n"].unique())
    fig, axes = plt.subplots(
        2,
        len(qubits),
        figsize=(4.35 * len(qubits), 7.4),
        sharex=False,
        constrained_layout=False,
    )
    if len(qubits) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    row_specs = [
        ("quantum", "Quantum / Heisenberg", "Logistic loss"),
        ("classical", "Classical / FCIM", "Logistic loss"),
    ]

    for column, n in enumerate(qubits):
        for row, (model, row_title, ylabel) in enumerate(row_specs):
            ax = axes[row, column]
            subset = curves[(curves["n"] == n) & (curves["model"] == model)].copy()
            for method in [
                "original_dense",
                "original_dense_probe",
                "penny_exact_aggregate",
                "penny_chebyshev_matrix_free",
                "penny_diagonal_fcim",
            ]:
                method_data = subset[subset["method"] == method]
                if method_data.empty:
                    continue
                method_data = method_data.sort_values("epoch")
                ax.plot(
                    method_data["epoch"],
                    method_data["loss"],
                    METHOD_STYLES.get(method, "-"),
                    color=METHOD_COLORS.get(method, None),
                    linewidth=1.6,
                    label=METHOD_LABELS.get(method, method),
                )
            finalize_axis(
                ax,
                title=f"{n} qubits — {row_title}",
                ylabel=ylabel if column == 0 else None,
            )
            ax.set_xlabel("Epoch")
            if (subset["method"] == "original_dense_probe").any():
                ax.text(
                    0.03,
                    0.05,
                    "dense original probed only",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="#555555",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "#dddddd",
                        "alpha": 0.8,
                        "pad": 2,
                    },
                )

    legend_items = {}
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)
    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=4,
            fontsize=9,
        )
    fig.suptitle(
        "Log-loss training curves: original dense notebook vs optimized PennyLane",
        fontsize=12,
        y=0.99,
    )
    fig.subplots_adjust(top=0.76, hspace=0.42, wspace=0.22)

    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIRECTORY / f"paper_training_curves_{suffix}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_efficiency_summary(summary: pd.DataFrame, suffix: str) -> Path:
    qubits = sorted(summary["n"].unique())
    quantum = summary[summary["model"] == "quantum"].copy()
    quantum["estimated_total_seconds"] = safe_numeric(
        quantum["estimated_total_seconds"]
    )
    quantum["static_bytes"] = safe_numeric(quantum["static_bytes"])
    quantum["loss_error_vs_dense"] = safe_numeric(quantum["loss_error_vs_dense"])
    quantum["gradient_error_vs_dense"] = safe_numeric(
        quantum["gradient_error_vs_dense"]
    )

    methods = [
        "original_dense",
        "original_dense_probe",
        "penny_exact_aggregate",
        "penny_chebyshev_matrix_free",
    ]
    x = np.arange(len(qubits))
    width = 0.2

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    ax_time, ax_speed, ax_memory, ax_error = axes.ravel()

    for offset_index, method in enumerate(methods):
        rows = []
        for n in qubits:
            method_rows = quantum[
                (quantum["n"] == n) & (quantum["method"] == method)
            ]
            rows.append(method_rows.iloc[0] if not method_rows.empty else None)
        values = [
            np.nan if row is None else row["estimated_total_seconds"] for row in rows
        ]
        if np.all(np.isnan(values)):
            continue
        offset = (offset_index - 1.5) * width
        ax_time.bar(
            x + offset,
            values,
            width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            alpha=0.9,
        )

    ax_time.set_yscale("log")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([str(n) for n in qubits])
    ax_time.set_xlabel("Qubits")
    finalize_axis(
        ax_time,
        title="Estimated quantum training time",
        ylabel="Seconds, log scale",
    )

    for method in ["penny_exact_aggregate", "penny_chebyshev_matrix_free"]:
        speedups = []
        for n in qubits:
            baseline = quantum[
                (quantum["n"] == n)
                & (quantum["method"].isin(["original_dense", "original_dense_probe"]))
            ]
            candidate = quantum[(quantum["n"] == n) & (quantum["method"] == method)]
            if baseline.empty or candidate.empty:
                speedups.append(np.nan)
            else:
                speedups.append(
                    float(baseline.iloc[0]["estimated_total_seconds"])
                    / float(candidate.iloc[0]["estimated_total_seconds"])
                )
        ax_speed.plot(
            qubits,
            speedups,
            marker="o",
            linewidth=2.0,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
        )
    ax_speed.set_yscale("log")
    ax_speed.set_xlabel("Qubits")
    finalize_axis(ax_speed, title="Speedup vs original dense", ylabel="× faster")

    for method in methods:
        memory = []
        for n in qubits:
            method_rows = quantum[
                (quantum["n"] == n) & (quantum["method"] == method)
            ]
            memory.append(
                np.nan
                if method_rows.empty
                else float(method_rows.iloc[0]["static_bytes"]) / 1024**2
            )
        if np.all(np.isnan(memory)):
            continue
        ax_memory.plot(
            qubits,
            memory,
            marker="o",
            linewidth=2.0,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            linestyle=METHOD_STYLES.get(method, "-"),
        )
    ax_memory.set_yscale("log")
    ax_memory.set_xlabel("Qubits")
    finalize_axis(ax_memory, title="Static data/operator memory", ylabel="MiB")

    for metric, label, marker in [
        ("loss_error_vs_dense", "loss error", "o"),
        (
            "gradient_error_vs_dense",
            "gradient error",
            "s",
        ),
    ]:
        for method in ["penny_exact_aggregate", "penny_chebyshev_matrix_free"]:
            values = []
            for n in qubits:
                method_rows = quantum[
                    (quantum["n"] == n) & (quantum["method"] == method)
                ]
                if method_rows.empty:
                    values.append(np.nan)
                else:
                    value = method_rows.iloc[0][metric]
                    values.append(np.nan if pd.isna(value) else max(float(value), 1e-16))
            ax_error.plot(
                qubits,
                values,
                marker=marker,
                linewidth=1.8,
                label=f"{METHOD_LABELS[method]} {label}",
                color=METHOD_COLORS[method],
                linestyle="-" if marker == "o" else "--",
            )
    ax_error.set_yscale("log")
    ax_error.set_xlabel("Qubits")
    finalize_axis(
        ax_error,
        title="Numerical error vs dense reference",
        ylabel="Absolute error, log scale",
    )

    for ax in [ax_time, ax_speed, ax_memory, ax_error]:
        ax.legend(fontsize=8)

    fig.suptitle(
        "Efficiency and numerical agreement for the quantum log-loss model",
        fontsize=12,
        y=1.03,
    )
    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIRECTORY / f"paper_efficiency_comparison_{suffix}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sampling_efficiency(sampling: pd.DataFrame | None, suffix: str) -> Path | None:
    if sampling is None or sampling.empty:
        return None
    sampling = sampling.copy()
    sampling["pass_seconds"] = safe_numeric(sampling["pass_seconds"])
    sampling["static_bytes"] = safe_numeric(sampling["static_bytes"])
    qubits = sorted(sampling["n"].unique())

    fig, axes = plt.subplots(
        1,
        len(qubits),
        figsize=(4.25 * len(qubits), 3.8),
        constrained_layout=True,
        sharey=True,
    )
    if len(qubits) == 1:
        axes = [axes]

    for ax, n in zip(axes, qubits):
        subset = sampling[sampling["n"] == n].copy()
        for method in [
            "original_dense",
            "penny_exact_aggregate",
            "penny_chebyshev_matrix_free",
        ]:
            method_data = subset[subset["method"] == method].sort_values("samples")
            if method_data.empty:
                continue
            measured = method_data[method_data["measured"].astype(str) == "True"]
            estimated = method_data[method_data["measured"].astype(str) != "True"]
            if not measured.empty:
                ax.plot(
                    measured["samples"],
                    measured["pass_seconds"],
                    marker="o",
                    linewidth=1.8,
                    color=METHOD_COLORS.get(method, None),
                    linestyle=METHOD_STYLES.get(method, "-"),
                    label=METHOD_LABELS.get(method, method),
                )
            if not estimated.empty:
                ax.plot(
                    estimated["samples"],
                    estimated["pass_seconds"],
                    marker="x",
                    linewidth=1.4,
                    color=METHOD_COLORS.get(method, None),
                    linestyle=":",
                    label=f"{METHOD_LABELS.get(method, method)} estimated",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training samples")
        finalize_axis(
            ax,
            title=f"{n} qubits",
            ylabel="Loss+gradient pass seconds",
        )
        ax.legend(fontsize=7)

    fig.suptitle(
        "Sampling efficiency: one quantum loss/gradient pass vs sample count",
        fontsize=12,
        y=1.06,
    )
    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIRECTORY / f"paper_sampling_efficiency_{suffix}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot paper-style comparison figures from benchmark CSVs."
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        default=[2, 4, 7],
        help="Qubit suffix to load, matching benchmark_paper_comparison.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = "_".join(str(n) for n in args.qubits)
    curves, summary, sampling = load_data(suffix)
    outputs = [
        plot_training_curves(curves, suffix),
        plot_efficiency_summary(summary, suffix),
    ]
    sampling_output = plot_sampling_efficiency(sampling, suffix)
    if sampling_output is not None:
        outputs.append(sampling_output)
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
