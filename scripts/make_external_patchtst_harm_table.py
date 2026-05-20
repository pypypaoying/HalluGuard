from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


PATCHTST_ROWS = [
    ("ETTh1", "96", 0.411, 0.025, 56.2, 25.0, "Yes"),
    ("ETTh1", "192", 0.105, -0.027, 75.0, 46.9, "Yes"),
    ("ETTh1", "336", 0.088, -0.024, 78.1, 50.0, "Yes"),
    ("ETTh1", "720", -0.217, -0.021, 75.0, 50.0, "No"),
    ("ETTm1", "96", -0.465, -0.354, 21.9, 12.5, "No"),
    ("ETTm1", "192", -0.007, -0.126, 18.8, 6.2, "No"),
    ("ETTm1", "336", -0.024, -0.250, 15.6, 6.2, "No"),
    ("ETTm1", "720", 0.144, -0.037, 12.5, 9.4, "Yes"),
]

SUMMARY_ROWS = [
    ("PatchTST avg.", "8 configs", 0.004, -0.105, 44.9, 25.8, "4/8"),
    ("DLinear avg.", "8 configs", -1.865, -0.391, 44.5, 22.7, "0/8"),
]


def draw_table() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 6.2), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    left, right = 0.03, 0.97
    top, bottom = 0.84, 0.115
    width = right - left
    height = top - bottom

    col_fracs = [0.13, 0.11, 0.17, 0.18, 0.14, 0.14, 0.13]
    xs = [left]
    for frac in col_fracs:
        xs.append(xs[-1] + frac * width)

    rows = PATCHTST_ROWS + SUMMARY_ROWS
    header_h = 0.15 * height
    row_h = (height - header_h) / len(rows)
    header_bottom = top - header_h

    ax.text(
        left,
        0.965,
        "External Fixture PatchTST Harm Diagnostic",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        fontfamily="DejaVu Serif",
        color="#111111",
    )
    ax.text(
        left,
        0.925,
        "Adaptive router on external forecasts; MSE delta is percentage vs. no correction. Positive values indicate harm.",
        ha="left",
        va="top",
        fontsize=10.5,
        fontfamily="DejaVu Serif",
        color="#555555",
    )

    ax.plot([left, right], [top, top], color="black", lw=1.35)
    ax.plot([left, right], [header_bottom, header_bottom], color="black", lw=1.1)
    ax.plot([left, right], [bottom, bottom], color="black", lw=1.35)
    for x in xs:
        ax.plot([x, x], [bottom, top], color="black", lw=0.72)

    headers = [
        "Dataset",
        "Horizon",
        "Router\nMSE Δ (%)",
        "Matched\nsmoothing Δ (%)",
        "Correction\nrate (%)",
        "Smoothing\naction (%)",
        "Harmed",
    ]
    for i, label in enumerate(headers):
        ax.text(
            (xs[i] + xs[i + 1]) / 2,
            (top + header_bottom) / 2,
            label,
            ha="center",
            va="center",
            fontsize=11.4,
            fontweight="bold",
            fontfamily="DejaVu Serif",
            color="black",
            linespacing=1.05,
        )

    red = "#e31a1c"
    black = "#111111"
    grey = "#555555"

    for ri, row in enumerate(rows):
        y_top = header_bottom - ri * row_h
        y_bottom = y_top - row_h
        y = (y_top + y_bottom) / 2

        if ri == len(PATCHTST_ROWS):
            ax.plot([left, right], [y_top, y_top], color="black", lw=1.05)
        ax.plot([left, right], [y_bottom, y_bottom], color="black", lw=0.42)

        dataset, horizon, router, matched, correction, smoothing, harmed = row
        values = [
            fill(dataset, 12),
            horizon,
            f"{router:.3f}",
            f"{matched:.3f}",
            f"{correction:.1f}",
            f"{smoothing:.1f}",
            harmed,
        ]

        for ci, value in enumerate(values):
            x = (xs[ci] + xs[ci + 1]) / 2
            color = black
            weight = "normal"
            if ci == 2:
                color = red if router > 0 else black
                weight = "bold"
            elif ci == 3:
                color = red if matched > 0 else black
            elif ci == 6:
                color = red if harmed in {"Yes", "4/8"} else black
                weight = "bold"
            elif ri >= len(PATCHTST_ROWS) and ci in {0, 2, 6}:
                weight = "bold"
            ax.text(
                x,
                y,
                value,
                ha="center",
                va="center",
                fontsize=10.8,
                fontweight=weight,
                fontfamily="DejaVu Serif",
                color=color,
            )

    ax.text(
        left,
        0.058,
        "Setup: external fixture = 2 datasets (ETTh1, ETTm1) x 2 backbones (DLinear, PatchTST) x 4 horizons "
        "(96, 192, 336, 720), 32 samples per config. Matched smoothing uses the same correction rate as the router.",
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontfamily="DejaVu Serif",
        color=grey,
    )
    ax.text(
        left,
        0.032,
        "Interpretation: PatchTST shows weak average improvement and 4/8 harmed configs, while DLinear improves on average with 0/8 harmed configs.",
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontfamily="DejaVu Serif",
        color=grey,
    )

    out = FIG_DIR / "external_patchtst_harm_table"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    draw_table()
    print(FIG_DIR / "external_patchtst_harm_table.png")
