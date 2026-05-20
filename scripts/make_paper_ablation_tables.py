from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ["96", "192", "336", "720", "Avg"]


FIG1 = {
    "filename": "base_vs_full_smoothing_table",
    "caption": "Base HalluGuard underperforms the full smoothing baseline; Dynamics narrows the gap.",
    "methods": [
        {
            "name": "Full smoothing\nbaseline",
            "mse": [4.020, 5.967, 7.034, 10.266, 6.822],
            "mae": [1.477, 1.842, 2.028, 2.471, 1.954],
        },
        {
            "name": "Base\nHalluGuard",
            "mse": [4.104, 6.110, 7.136, 10.517, 6.967],
            "mae": [1.496, 1.866, 2.044, 2.497, 1.976],
        },
        {
            "name": "Dynamics\nversion",
            "mse": [4.066, 6.056, 7.114, 10.503, 6.935],
            "mae": [1.485, 1.852, 2.038, 2.494, 1.967],
        },
    ],
}


FIG2 = {
    "filename": "router_vs_base_matched_table",
    "caption": "Router augmentation improves over Base HalluGuard, Dynamics, and the matched smoothing control.",
    "methods": [
        {
            "name": "Matched\nsmoothing\nbaseline",
            "mse": [4.073, 6.061, 7.093, 10.464, 6.923],
            "mae": [1.489, 1.858, 2.037, 2.491, 1.969],
        },
        {
            "name": "Base\nHalluGuard",
            "mse": [4.104, 6.110, 7.136, 10.517, 6.967],
            "mae": [1.496, 1.866, 2.044, 2.497, 1.976],
        },
        {
            "name": "Dynamics\nversion",
            "mse": [4.066, 6.056, 7.114, 10.503, 6.935],
            "mae": [1.485, 1.852, 2.038, 2.494, 1.967],
        },
        {
            "name": "Base +\nRouter",
            "mse": [4.056, 6.000, 7.072, 10.325, 6.863],
            "mae": [1.483, 1.845, 2.032, 2.478, 1.960],
        },
    ],
}


def draw_table(spec):
    n_methods = len(spec["methods"])
    row_h = 0.82
    header_h = 0.58
    caption_h = 0.34
    width = 9.2
    height = caption_h + header_h + n_methods * 2 * row_h + 0.12

    fig, ax = plt.subplots(figsize=(width, height), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    table_left = 0.035
    table_right = 0.965
    table_top = 0.90
    table_bottom = 0.08
    table_w = table_right - table_left
    table_h = table_top - table_bottom

    col_fracs = [0.24, 0.12, 0.125, 0.125, 0.125, 0.125, 0.14]
    xs = [table_left]
    for frac in col_fracs:
        xs.append(xs[-1] + frac * table_w)

    header_abs = 0.17 * table_h
    body_h = table_h - header_abs
    subrow_h = body_h / (n_methods * 2)
    y_header_bottom = table_top - header_abs

    # Caption.
    ax.text(
        table_left,
        0.965,
        spec["caption"],
        ha="left",
        va="top",
        fontsize=12.5,
        fontfamily="DejaVu Serif",
        color="#222222",
    )

    # Grid border and header rules.
    ax.plot([table_left, table_right], [table_top, table_top], color="black", lw=1.2)
    ax.plot([table_left, table_right], [y_header_bottom, y_header_bottom], color="black", lw=1.15)
    ax.plot([table_left, table_right], [table_bottom, table_bottom], color="black", lw=1.2)
    for x in xs:
        ax.plot([x, x], [table_bottom, table_top], color="black", lw=0.8)

    headers = ["Methods", "Metric"] + HORIZONS
    for i, label in enumerate(headers):
        ax.text(
            (xs[i] + xs[i + 1]) / 2,
            (table_top + y_header_bottom) / 2,
            label,
            ha="center",
            va="center",
            fontsize=14.5,
            fontweight="bold",
            fontfamily="DejaVu Serif",
            color="black",
        )

    num_color = "#ff1f1f"
    body_top = y_header_bottom

    for mi, method in enumerate(spec["methods"]):
        group_top = body_top - mi * 2 * subrow_h
        group_mid = group_top - subrow_h
        group_bottom = group_top - 2 * subrow_h

        # Horizontal group separator and metric separator.
        ax.plot([table_left, table_right], [group_bottom, group_bottom], color="black", lw=0.75)
        ax.plot([xs[1], table_right], [group_mid, group_mid], color="black", lw=0.45)

        ax.text(
            (xs[0] + xs[1]) / 2,
            (group_top + group_bottom) / 2,
            method["name"],
            ha="center",
            va="center",
            fontsize=12.6,
            fontstyle="italic",
            fontfamily="DejaVu Serif",
            color="black",
            linespacing=1.15,
        )

        for row_name, values, y0, y1 in [
            ("MSE", method["mse"], group_top, group_mid),
            ("MAE", method["mae"], group_mid, group_bottom),
        ]:
            ax.text(
                (xs[1] + xs[2]) / 2,
                (y0 + y1) / 2,
                row_name,
                ha="center",
                va="center",
                fontsize=13.2,
                fontweight="bold",
                fontfamily="DejaVu Serif",
                color="black",
            )
            for vi, value in enumerate(values):
                ci = vi + 2
                ax.text(
                    (xs[ci] + xs[ci + 1]) / 2,
                    (y0 + y1) / 2,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=13.4,
                    fontweight="bold",
                    fontfamily="DejaVu Serif",
                    color=num_color,
                )

    ax.text(
        table_left,
        0.028,
        "Clean benchmark; values are averaged over datasets and forecasting backbones. Lower MSE/MAE is better.",
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontfamily="DejaVu Serif",
        color="#555555",
    )

    out_base = FIG_DIR / spec["filename"]
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw_table(FIG1)
    draw_table(FIG2)
    print(FIG_DIR / "base_vs_full_smoothing_table.png")
    print(FIG_DIR / "router_vs_base_matched_table.png")
