from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


rows = [
    (
        "Mechanism",
        "Fixed HalluGuard correction calibrated on validation data",
        "Validation-only router selects the correction action per sample",
    ),
    (
        "Clean mean MSE delta",
        "-0.623%",
        "-1.289%",
    ),
    (
        "Improved clean configs",
        "15 / 16",
        "15 / 16",
    ),
    (
        "Control evidence",
        "Beats random controls 15 / 16; paired win rate 0.9375",
        "Beats random action 15 / 16; paired win rate 0.9500",
    ),
    (
        "Stress evidence",
        "Boundary stress: -0.932%",
        "Stress mean: -1.391%; boundary stress: -1.483%",
    ),
    (
        "Action space",
        "Single calibrated repair policy",
        "No correction, boundary repair, dynamics repair, and smoothing actions",
    ),
    (
        "Main finding",
        "Boundary-aware local dynamics gives a stable post-processing signal",
        "Routing improves the correction by matching actions to forecast shapes",
    ),
    (
        "Main limitation",
        "Fixed correction cannot adapt to heterogeneous forecast errors",
        "External PatchTST harm remains visible in the compact diagnostic fixture",
    ),
]


fig = plt.figure(figsize=(10.8, 5.9), dpi=220)
ax = plt.axes([0, 0, 1, 1])
ax.set_axis_off()

bg = "#fbfbfa"
ink = "#202225"
muted = "#5b6470"
rule = "#b9c0c8"
header_bg = "#e8edf3"
v1_bg = "#f6f8fb"
v2_bg = "#f2f7f3"
accent = "#2f5d8c"
v2_accent = "#4f7d57"

fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

ax.text(
    0.055,
    0.945,
    "Base HalluGuard vs. Router-Augmented HalluGuard",
    fontsize=18,
    fontweight="bold",
    color=ink,
    ha="left",
    va="top",
)
ax.text(
    0.055,
    0.902,
    "MSE delta is measured against the uncorrected forecast; lower values are better.",
    fontsize=9.5,
    color=muted,
    ha="left",
    va="top",
)

left = 0.055
right = 0.945
top = 0.845
bottom = 0.105
width = right - left
height = top - bottom

col_w = [0.24, 0.36, 0.40]
x = [left]
for w in col_w[:-1]:
    x.append(x[-1] + w * width)
x.append(right)

header_h = 0.085
row_h = (height - header_h) / len(rows)

# Header and subtle column fills.
ax.add_patch(Rectangle((left, top - header_h), width, header_h, facecolor=header_bg, edgecolor="none"))
ax.add_patch(Rectangle((x[1], bottom), x[2] - x[1], height - header_h, facecolor=v1_bg, edgecolor="none"))
ax.add_patch(Rectangle((x[2], bottom), x[3] - x[2], height - header_h, facecolor=v2_bg, edgecolor="none"))

headers = ["Evaluation item", "Version 1: Base HalluGuard", "Version 2: + Router"]
for i, h in enumerate(headers):
    ax.text(
        x[i] + 0.012,
        top - header_h / 2,
        h,
        fontsize=10.3,
        fontweight="bold",
        color=accent if i < 2 else v2_accent,
        ha="left",
        va="center",
    )

# Rules.
ax.plot([left, right], [top, top], color=ink, lw=1.2)
ax.plot([left, right], [top - header_h, top - header_h], color=rule, lw=0.85)
ax.plot([left, right], [bottom, bottom], color=ink, lw=1.2)
for xi in x[1:-1]:
    ax.plot([xi, xi], [bottom, top], color="#d8dde3", lw=0.7)

for r, row in enumerate(rows):
    y_top = top - header_h - r * row_h
    y_mid = y_top - row_h / 2
    y_bot = y_top - row_h
    ax.plot([left, right], [y_bot, y_bot], color="#e1e5ea", lw=0.65)

    item, v1, v2 = row
    ax.text(
        x[0] + 0.012,
        y_mid,
        wrap(item, 21),
        fontsize=9.2,
        fontweight="bold",
        color=ink,
        ha="left",
        va="center",
    )

    v1_weight = "bold" if item in {"Clean mean MSE delta", "Improved clean configs"} else "normal"
    v2_weight = "bold" if item in {"Clean mean MSE delta", "Improved clean configs", "Control evidence"} else "normal"
    ax.text(
        x[1] + 0.014,
        y_mid,
        wrap(v1, 34),
        fontsize=9.0,
        fontweight=v1_weight,
        color=ink,
        ha="left",
        va="center",
        linespacing=1.22,
    )
    ax.text(
        x[2] + 0.014,
        y_mid,
        wrap(v2, 39),
        fontsize=9.0,
        fontweight=v2_weight,
        color=ink,
        ha="left",
        va="center",
        linespacing=1.22,
    )

ax.text(
    left,
    0.060,
    "Source: local prototype experiments summarized in docs/research_narrative.zh-CN.md. "
    "The table reports preliminary evidence, not final benchmark claims.",
    fontsize=8.2,
    color=muted,
    ha="left",
    va="bottom",
)

png_path = FIG_DIR / "version1_version2_results_table.png"
pdf_path = FIG_DIR / "version1_version2_results_table.pdf"
svg_path = FIG_DIR / "version1_version2_results_table.svg"
fig.savefig(png_path, dpi=300, facecolor=bg, bbox_inches="tight", pad_inches=0.14)
fig.savefig(pdf_path, facecolor=bg, bbox_inches="tight", pad_inches=0.14)
fig.savefig(svg_path, facecolor=bg, bbox_inches="tight", pad_inches=0.14)
print(png_path)
print(pdf_path)
print(svg_path)
