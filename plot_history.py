"""Plot how the errors evolve during training.

Reads models/history.csv (final model, 81 sessions, RTX 5070) and
models/history_cpu_16sessions.csv (first model, 16 sessions, CPU), and writes:
    figures/turn_error.png     error in turns on validation (main metric)
    figures/overall_error.png  overall error on validation
    figures/overfitting.png    training error vs validation error (final model)

Usage: .venv/bin/python plot_history.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (categorical slots 1 and 2, light mode) + text inks
SURFACE, TEXT, TEXT_2, GRID = "#ffffff", "#191919", "#666666", "#e6e6e6"
BLUE, ORANGE, NEUTRAL = "#257de0", "#f47230", "#8c8c8c"

# (file, legend, color, epoch of the kept model)
FINAL = ("models/history.csv", "Final model (81 sessions, RTX 5070)", BLUE, 26)
# the first model was selected on the overall error at the time -> epoch 4
FIRST = ("models/history_cpu_16sessions.csv", "First model (16 sessions, CPU)", ORANGE, 4)
BASELINE = {"val_mae_turns": 0.1806, "val_mae": 0.0307}  # "always straight ahead"

OUT = Path("figures")


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0, pad=6)
    ax.set_xlabel("Epoch", color=TEXT_2, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10, labelpad=8)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left", pad=14, fontweight="bold")


def plot_metric(metric, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax, title, ylabel)
    decimals = 3 if metric == "val_mae_turns" else 4

    for path, label, color, kept in (FINAL, FIRST):
        h = load(path)
        x, y = h["epoch"], h[metric]
        ax.plot(x, y, color=color, linewidth=2, label=label, solid_capstyle="round", zorder=3)
        i = int(kept) - 1  # kept model
        ax.scatter([x[i]], [y[i]], s=48, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(f"{y[i]:.{decimals}f}",
                    (x[i], y[i]), xytext=(0, -16 if color == BLUE else 10), textcoords="offset points",
                    ha="center", fontsize=9, color=TEXT)

    ref = BASELINE[metric]
    ax.axhline(ref, color=NEUTRAL, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.text(30, ref, f'"Always straight ahead" baseline: {ref:.{decimals}f}',
            ha="right", va="bottom", fontsize=9, color=TEXT_2)

    ax.set_xlim(0.5, 30.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9), frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.text(0.01, 0.01, "Lower is better. Dot = kept model (epoch 26 for the final model, epoch 4 for the first).",
             fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / filename, facecolor=SURFACE)
    plt.close(fig)


def plot_overfitting():
    h = load(FINAL[0])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    panels = (("train_loss", "Training error", "Loss (Smooth L1)", BLUE),
              ("val_mae_turns", "Validation error (turns)", "MAE in turns", BLUE))
    for ax, (key, title, ylabel, color) in zip(axes, panels):
        style(ax, title, ylabel)
        ax.plot(h["epoch"], h[key], color=color, linewidth=2, solid_capstyle="round")
        ax.set_xticks([1, 10, 20, 30])
    axes[1].axhline(BASELINE["val_mae_turns"], color=NEUTRAL, linewidth=1.5, linestyle=(0, (4, 3)))
    axes[1].text(30, BASELINE["val_mae_turns"], "baseline 0.181", ha="right", va="bottom",
                 fontsize=8, color=TEXT_2)
    fig.text(0.01, 0.01, "Final model. Training error keeps falling while validation error levels off "
             "around 0.137: past that point the model learns nothing that generalizes.",
             fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "overfitting.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plot_metric("val_mae_turns", "Error in turns on validation", "MAE in turns", "turn_error.png")
    plot_metric("val_mae", "Overall error on validation", "MAE (all frames)", "overall_error.png")
    plot_overfitting()
    print("wrote:", ", ".join(str(p) for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
