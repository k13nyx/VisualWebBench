"""
Scatterplot of overall score vs. average cost per 1M (in USD), from summary.xlsx.

Usage:
    python visualize.py [--summary path/to/summary.xlsx] [--output output.png]
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


p = argparse.ArgumentParser(description="Scatterplot of overall score vs. average cost per 1M tokens (USD).")
p.add_argument("--summary", required=True, help="Path to summary.xlsx")
p.add_argument("--output", "-o", required=True, help="Output PNG path")
args = p.parse_args()


XLSX = args.summary
OUT = args.output

X_COL, Y_COL = "avg_cost_per_1M", "overall_score"
X_LABEL, Y_LABEL = "Avg. cost per 1M (in USD)", "Overall Score"
MIN_RELIABILITY = 90          # models below this are left out; set to 0 to keep all
LOG_X = True                  # cost spans a wide range, so log scale reads better

X_TICKS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

FRONTIER_COLOR = "#2a78d6"    # best score at its price
OTHER_COLOR = "#8f8d86"       # everything else
TEXT_COLOR = "#52514e"

# Label position per model: (dx, dy) in points from the dot, and horizontal alignment.
# Models not listed here get the default: to the right of the dot.
DEFAULT_LABEL = (8, 0, "left")
LABELS = {
    "qwen3.8-flash":       (0, 10, "center"),
    "mimo-v2.6-flash":     (-8, 0, "right"),
    "claude-haiku-5.5":    (-8, 0, "right"),
    "deepseek-v4.1-flash": (0, 10, "center"),
    "mimo-v2.6-pro":       (-8, 0, "right"),
}


df = pd.read_excel(XLSX)

# calculate overall score by averaging the scores of VisualWebBench tasks
task_cols = [
    "Captioning", "WebQA", "Heading OCR",
    "Element OCR", "Element Grounding",
    "Action Prediction", "Action Grounding"
]
df["overall_score"] = df[task_cols].mean(axis=1)    # mean skips missing values by default

# calculate model reliability by computing % of items that produced a non-empty answers
df["reliability"] = ((df["n_items"] - df["n_empty"]) / df["n_items"]) * 100


# keep only rows whose reliability ≥ MIN_RELIABILITY, then sort them by cost
df = df[df["reliability"] >= MIN_RELIABILITY].sort_values(X_COL).reset_index(drop=True)

# Pareto frontier: walking up in cost, keep each model that beats every cheaper one.
df["frontier"] = df[Y_COL] > df[Y_COL].cummax().shift(fill_value=float("-inf"))

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

front = df[df["frontier"]].sort_values(X_COL)
other = df[~df["frontier"]]
ax.plot(front[X_COL], front[Y_COL], c=FRONTIER_COLOR, alpha=0.35, lw=2, zorder=1)
ax.scatter(front[X_COL], front[Y_COL], c=FRONTIER_COLOR, s=70, label="Frontier", linewidths=1.5, zorder=3)
ax.scatter(other[X_COL], other[Y_COL], c=OTHER_COLOR, s=70, label="Other", linewidths=1.5, zorder=3)

for _, r in df.iterrows():
    dx, dy, ha = LABELS.get(r["model"], DEFAULT_LABEL)
    ax.annotate(
        r["model"], (r[X_COL], r[Y_COL]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha, va="center", fontsize=8,
        color="#0b0b0b" if r["frontier"] else TEXT_COLOR,
        fontweight="bold" if r["frontier"] else "normal"
    )

if LOG_X:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

ax.grid(True, color="#e1e0d9", lw=1)
ax.set_axisbelow(True)

ax.set_xlabel(X_LABEL + (" (log scale)" if LOG_X else ""), color=TEXT_COLOR)
ax.set_ylabel(Y_LABEL, color=TEXT_COLOR)
ax.set_title("Overall score vs. average cost per $1M", loc="left", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", frameon=False, fontsize=9)


fig.tight_layout()
fig.savefig(OUT)
print(f"Saved {OUT}")