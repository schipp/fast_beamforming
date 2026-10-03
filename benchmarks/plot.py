"""Figures for the README from results/benchmark.json and results/compare.json.

    python benchmarks/plot.py      # writes results/implementations.png and results/software.png

Colours: the sequence of Petroff (2021), arXiv:2107.02270, made to be told apart with any colour vision.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results"

# label, colour
IMPLEMENTATIONS = {
    "CPU": {
        "numpy_einsum": ("numpy, einsum", "#94a4a2"),
        "numpy_matmul": ("numpy", "#3f90da"),
        "torch_matmul": ("pytorch", "#ffa90e"),
        "jax_matmul_cpu": ("jax", "#bd1f01"),
        "numba_loops": ("numba", "#832db6"),
    },
    "GPU": {
        "torch_matmul_cuda": ("pytorch", "#ffa90e"),
        "cupy_matmul": ("cupy", "#a96b59"),
        "jax_matmul_gpu": ("jax", "#bd1f01"),
    },
}
METHODS = {"crosscorrelation": "cross-correlation, Bartlett", "mvdr": "MVDR", "music": "MUSIC"}


SOFTWARE = {
    "plane waves": {
        "obspy": ("ObsPy array_processing", "#3f90da"),
        "twistpy": ("TwistPy BeamformingArray", "#bd1f01"),
        "pytorch_planewave": ("pytorch (notebook 3)", "#ffa90e"),
    },
    "sources on a grid": {
        "acoular": ("acoular BeamformerBase", "#832db6"),
        "covseisnet": ("covseisnet Beam", "#a96b59"),
        "beampower": ("beampower", "#94a4a2"),
        "pytorch_mfp": ("pytorch (notebook 3)", "#ffa90e"),
    },
}


def duration(seconds):
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.0f} h"


def legend(axs_row):
    """One legend per row, with the entries of all its panels."""
    entries = {}
    for ax in axs_row:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    axs_row[-1].legend(entries.values(), entries.keys(), fontsize=8, loc="center left",
                       bbox_to_anchor=(1.02, 0.5), frameon=False)  # fmt: skip


def style(ax, xlabel=True):
    ax.set(xscale="log", yscale="log")
    ax.grid(True, which="major", color="#e4e3df", lw=0.8)
    if xlabel:
        ax.set_xlabel("number of sensors")


def plot_implementations(window, filename):
    runs = [
        r
        for r in json.loads((RESULTS / "benchmark.json").read_text())["runs"]
        if r["window_length"] == window
    ]
    fig, axs = plt.subplots(2, len(METHODS), figsize=(4.2 * len(METHODS), 8), sharey=True, squeeze=False)
    for row, (hardware, cases) in enumerate(IMPLEMENTATIONS.items()):
        for col, (method, method_label) in enumerate(METHODS.items()):
            ax = axs[row, col]
            if hardware == "GPU":  # pytorch on the CPU, for reference
                data = sorted((r["n_sensors"], r["runtime"]) for r in runs
                              if r["case"] == "torch_matmul" and r["method"] == method)  # fmt: skip
                if data:
                    ax.plot(*zip(*data), c="#ffa90e", lw=5, alpha=0.3, label="pytorch, CPU", zorder=0)
            for case, (label, color) in cases.items():
                data = sorted((r["n_sensors"], r["runtime"]) for r in runs
                              if r["case"] == case and r["method"] == method)  # fmt: skip
                if data:
                    ax.plot(*zip(*data), c=color, lw=2, marker="o", ms=4, mec="w", label=label)
            style(ax, xlabel=row == 1)
            ax.set_title(f"{hardware}: {method_label}")
        axs[row, 0].set_ylabel("seconds")
        legend(axs[row])
    n_freqs = runs[0]["n_freqs"]
    fig.suptitle(f"From spectra to beampowers: 10 000 grid points, 100 time windows of {duration(window)}, "
                 f"0.1–1 Hz ({n_freqs} frequencies)")  # fmt: skip
    fig.savefig(RESULTS / filename, dpi=110, bbox_inches="tight")


def plot_software():
    runs = [r for r in json.loads((RESULTS / "compare.json").read_text())["runs"] if "error" not in r]
    windows = sorted({r["window_samples"] for r in runs})
    fig, axs = plt.subplots(2, len(windows), figsize=(4.2 * len(windows), 8), sharey=True, squeeze=False)
    for row, (problem, tools) in enumerate(SOFTWARE.items()):
        for col, window in enumerate(windows):
            ax = axs[row, col]
            for tool, (label, color) in tools.items():
                data = sorted((r["n_sensors"], r["seconds_per_window"]) for r in runs
                              if r["tool"] == tool and r["window_samples"] == window)  # fmt: skip
                if data:
                    ax.plot(*zip(*data), c=color, lw=2, marker="o", ms=4, mec="w", label=label)
            style(ax, xlabel=row == 1)
            ax.set_title(f"{problem}, {duration(window / 10)} windows")
        axs[row, 0].set_ylabel("seconds per window")
        legend(axs[row])
    fig.suptitle("Other beamforming software and the pytorch code of this repository, on the same recordings "
                 "(about 10 000 grid points)")  # fmt: skip
    fig.savefig(RESULTS / "software.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    if (RESULTS / "benchmark.json").exists():
        plot_implementations(600, "implementations.png")
    if (RESULTS / "compare.json").exists():
        plot_software()
