from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


ROWS = [
    {
        "experiment": "Boundary stress",
        "purpose": "Boundary jump repair",
        "main": -2.485757,
        "matched": -0.739298,
        "gain": -1.746459,
        "beats_matched": "16/16",
        "beats_random": "16/16",
        "takeaway": "Strong boundary-specific evidence",
    },
    {
        "experiment": "Slope break stress",
        "purpose": "Abrupt slope transition",
        "main": -2.203353,
        "matched": -0.718709,
        "gain": -1.484644,
        "beats_matched": "15/16",
        "beats_random": "16/16",
        "takeaway": "Robust to slope mismatch",
    },
    {
        "experiment": "Delayed level shift",
        "purpose": "Delayed horizontal offset",
        "main": -2.160671,
        "matched": -0.708956,
        "gain": -1.451715,
        "beats_matched": "16/16",
        "beats_random": "14/16",
        "takeaway": "Handles late level shift",
    },
    {
        "experiment": "High-frequency stress",
        "purpose": "Unsupported high-frequency perturbation",
        "main": -2.895440,
        "matched": -1.172253,
        "gain": -1.723187,
        "beats_matched": "16/16",
        "beats_random": "13/16",
        "takeaway": "Largest MSE reduction",
    },
    {
        "experiment": "Selected stress avg.",
        "purpose": "Four targeted stress families",
        "main": -2.436305,
        "matched": -0.834804,
        "gain": -1.601502,
        "beats_matched": "63/64",
        "beats_random": "59/64",
        "takeaway": "Not explained by equal-rate smoothing",
    },
    {
        "experiment": "Full stress suite",
        "purpose": "All six stress families",
        "main": -2.508625,
        "matched": -0.879944,
        "gain": -1.628681,
        "beats_matched": "95/96",
        "beats_random": "89/96",
        "takeaway": "General stress robustness",
    },
]


def draw_table() -> None:
    fig, ax = plt.subplots(figsize=(14.8, 5.6), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    left, right = 0.025, 0.975
    top, bottom = 0.865, 0.105
    width = right - left
    height = top - bottom

    col_fracs = [0.165, 0.205, 0.115, 0.135, 0.115, 0.105, 0.095, 0.165]
    xs = [left]
    for frac in col_fracs:
        xs.append(xs[-1] + frac * width)

    header_h = 0.155 * height
    row_h = (height - header_h) / len(ROWS)
    header_bottom = top - header_h

    ax.text(
        left,
        0.965,
        "Validation Stress Tests and Matched Smoothing Control",
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
        "Final HalluGuard mechanism; MSE delta is percentage vs. no correction. Lower is better.",
        ha="left",
        va="top",
        fontsize=10.5,
        fontfamily="DejaVu Serif",
        color="#555555",
    )

    # Table rules.
    ax.plot([left, right], [top, top], color="black", lw=1.35)
    ax.plot([left, right], [header_bottom, header_bottom], color="black", lw=1.1)
    ax.plot([left, right], [bottom, bottom], color="black", lw=1.35)
    for x in xs:
        ax.plot([x, x], [bottom, top], color="black", lw=0.72)

    headers = [
        "Experiment",
        "Purpose",
        "Final\nMSE Δ (%)",
        "Matched\nsmoothing Δ (%)",
        "Gain vs.\nmatched",
        "Beats\nmatched",
        "Beats\nrandom",
        "Takeaway",
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
    dark = "#111111"
    grey = "#444444"

    for ri, row in enumerate(ROWS):
        y_top = header_bottom - ri * row_h
        y_bottom = y_top - row_h
        y = (y_top + y_bottom) / 2
        if ri == len(ROWS) - 2:
            ax.plot([left, right], [y_top, y_top], color="black", lw=1.0)
        ax.plot([left, right], [y_bottom, y_bottom], color="black", lw=0.42)

        values = [
            fill(row["experiment"], 16),
            fill(row["purpose"], 25),
            f"{row['main']:.3f}",
            f"{row['matched']:.3f}",
            f"{row['gain']:.3f}",
            row["beats_matched"],
            row["beats_random"],
            fill(row["takeaway"], 22),
        ]

        for ci, value in enumerate(values):
            color = red if ci in {2, 3, 4} else dark
            weight = "bold" if ci in {0, 2, 4, 5} else "normal"
            align = "center" if ci not in {1, 7} else "left"
            x = (xs[ci] + xs[ci + 1]) / 2
            if align == "left":
                x = xs[ci] + 0.012 * width
            ax.text(
                x,
                y,
                value,
                ha=align,
                va="center",
                fontsize=10.6,
                fontweight=weight,
                fontfamily="DejaVu Serif",
                color=color,
                linespacing=1.12,
            )

    ax.text(
        left,
        0.052,
        "Matched smoothing control uses the same trigger/correction rate as the main method; "
        "beats columns count dataset-backbone-horizon configurations.",
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontfamily="DejaVu Serif",
        color=grey,
    )

    out = FIG_DIR / "validation_stress_results_table"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    draw_table()
    print(FIG_DIR / "validation_stress_results_table.png")
