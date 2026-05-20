from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


ROWS = [
    {
        "name": "Matched\nsmoothing\nbaseline",
        "clean": -0.722,
        "clean_patch": -0.291,
        "stress": -0.880,
        "external_patch": -0.102,
    },
    {
        "name": "Base\nHalluGuard",
        "clean": -0.058,
        "clean_patch": -0.046,
        "stress": -0.090,
        "external_patch": -0.004,
    },
    {
        "name": "Dynamics\nversion",
        "clean": -0.623,
        "clean_patch": -0.109,
        "stress": -0.644,
        "external_patch": -0.015,
    },
    {
        "name": "Base +\nRouter",
        "clean": -1.289,
        "clean_patch": -0.298,
        "stress": -1.391,
        "external_patch": 0.004,
    },
    {
        "name": "Final\nHalluGuard",
        "clean": -2.193,
        "clean_patch": -0.617,
        "stress": -2.509,
        "external_patch": -0.065,
    },
]


def fmt(value: float) -> str:
    return f"{value:+.3f}" if value > 0 else f"{value:.3f}"


def draw_table() -> None:
    n_rows = len(ROWS)
    fig, ax = plt.subplots(figsize=(12.2, 6.0), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    left, right = 0.04, 0.96
    top, bottom = 0.84, 0.125
    width = right - left
    height = top - bottom

    col_fracs = [0.25, 0.18, 0.20, 0.17, 0.20]
    xs = [left]
    for frac in col_fracs:
        xs.append(xs[-1] + frac * width)

    header_h = 0.20 * height
    row_h = (height - header_h) / n_rows
    header_bottom = top - header_h

    ax.text(
        left,
        0.965,
        "Final HalluGuard Delta Results",
        ha="left",
        va="top",
        fontsize=17.4,
        fontweight="bold",
        fontfamily="DejaVu Serif",
        color="#111111",
    )
    ax.text(
        left,
        0.922,
        "MSE delta percentage vs. no correction. Lower is better; positive values indicate harm.",
        ha="left",
        va="top",
        fontsize=10.8,
        fontfamily="DejaVu Serif",
        color="#555555",
    )

    ax.plot([left, right], [top, top], color="black", lw=1.35)
    ax.plot([left, right], [header_bottom, header_bottom], color="black", lw=1.15)
    ax.plot([left, right], [bottom, bottom], color="black", lw=1.35)
    for x in xs:
        ax.plot([x, x], [bottom, top], color="black", lw=0.78)

    headers = [
        "Methods",
        "Clean\nMSE delta",
        "Clean PatchTST\nMSE delta",
        "Stress\nMSE delta",
        "External PatchTST\nMSE delta",
    ]
    for i, label in enumerate(headers):
        ax.text(
            (xs[i] + xs[i + 1]) / 2,
            (top + header_bottom) / 2,
            label,
            ha="center",
            va="center",
            fontsize=12.2,
            fontweight="bold",
            fontfamily="DejaVu Serif",
            color="black",
            linespacing=1.08,
        )

    num_color = "#ff1f1f"
    harm_color = "#c40000"

    for ri, row in enumerate(ROWS):
        y_top = header_bottom - ri * row_h
        y_bottom = y_top - row_h
        y = (y_top + y_bottom) / 2
        ax.plot([left, right], [y_bottom, y_bottom], color="black", lw=0.55)

        ax.text(
            (xs[0] + xs[1]) / 2,
            y,
            row["name"],
            ha="center",
            va="center",
            fontsize=12.8,
            fontstyle="italic",
            fontfamily="DejaVu Serif",
            color="black",
            linespacing=1.12,
        )

        values = [row["clean"], row["clean_patch"], row["stress"], row["external_patch"]]
        for vi, value in enumerate(values):
            ci = vi + 1
            is_harm = value > 0
            ax.text(
                (xs[ci] + xs[ci + 1]) / 2,
                y,
                fmt(value),
                ha="center",
                va="center",
                fontsize=13.5,
                fontweight="bold",
                fontfamily="DejaVu Serif",
                color=harm_color if is_harm else num_color,
            )

    ax.text(
        left,
        0.070,
        "Clean: 16 dataset-backbone-horizon configs; Clean/External PatchTST: 8 PatchTST configs; Stress: 96 stress configs.",
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontfamily="DejaVu Serif",
        color="#555555",
    )
    ax.text(
        left,
        0.043,
        "External fixture is used as a harm diagnostic, not as a broad external-generalization claim.",
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontfamily="DejaVu Serif",
        color="#555555",
    )

    out = FIG_DIR / "final_delta_results_table"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    draw_table()
    print(FIG_DIR / "final_delta_results_table.png")
