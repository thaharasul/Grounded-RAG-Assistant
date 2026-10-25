"""Regenerate the results summary chart from the committed eval judgments.

Reads ``data/eval_summary.csv`` (the hand grading of each probe, for both the
300-word and 175-word chunking runs) and writes ``data/eval_summary.png``. The
chart mirrors the results table in the README and makes the headline finding
visible: removing embedding truncation moved answers from ungrounded to
grounded without improving end-to-end accuracy.

Run:
    python src/plot_results.py
"""

import csv
import os
from collections import Counter

import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "eval_summary.csv")
PNG_PATH = os.path.join(DATA_DIR, "eval_summary.png")

# Fixed display order (best to worst) so the bars always tell the same story.
OUTCOMES = [
    ("grounded_correct", "Grounded and correct"),
    ("grounded_partial", "Grounded, with an error"),
    ("correct_ungrounded", "Correct but ungrounded\n(answered from model weights)"),
    ("wrong", "Wrong"),
]

# Runs in chronological order. Categorical hues assigned in fixed order and never
# cycled; the pair is validated for colourblind separation against the surface
# below (worst-pair CVD dE 24.7, normal-vision 33.6, contrast >= 3:1).
RUNS = [
    ("300w", "300-word chunks — 79% of chunks truncated", "#2a78d6"),
    ("175w", "175-word chunks — 4% of chunks truncated", "#eb6834"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"


def load_counts():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    per_run = {
        run: Counter(r["outcome"] for r in rows if r["run"] == run)
        for run, _, _ in RUNS
    }
    return per_run, len({r["id"] for r in rows})


def main():
    per_run, n_probes = load_counts()

    labels = [label for _, label in OUTCOMES]
    y = list(range(len(OUTCOMES)))
    height = 0.38  # paired bars, with a surface gap left between them

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (run, run_label, color) in enumerate(RUNS):
        counts = per_run[run]
        values = [counts.get(key, 0) for key, _ in OUTCOMES]
        offset = (i - 0.5) * height  # earlier run sits on top within each group
        bars = ax.barh(
            [pos + offset for pos in y],
            values,
            height=height * 0.92,
            color=color,
            label=run_label,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + 0.07,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                va="center",
                ha="left",
                fontsize=10,
                color=INK_MUTED,
                zorder=4,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.invert_yaxis()  # best outcome on top
    ax.set_xlim(0, 3.7)
    ax.set_xlabel("Number of probes", fontsize=10, color=INK_MUTED)

    fig.text(
        0.012,
        0.955,
        "Removing embedding truncation improved grounding, not accuracy",
        ha="left",
        va="top",
        fontsize=13,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.012,
        0.885,
        f"{n_probes} probes, hand-graded from the committed transcripts",
        ha="left",
        va="top",
        fontsize=10,
        color=INK_MUTED,
    )

    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, colors=INK_MUTED)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(PNG_PATH, dpi=150, facecolor=SURFACE)
    print(f"Wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
