"""
Per-position entropy grouped by KLIFS functional region (dot plot).

Every KLIFS position is drawn as one point, grouped by functional region, with a
horizontal bar marking the region mean. Within each region, points are placed
left -> right in KLIFS position order, so the lowest / highest positions can be
annotated with their KLIFS number.

Input : entropy_analysis_output/residue_property_entropy_merged.csv
        (produced by entropy_analysis.ipynb, 482 human kinases)
Output: entropy_analysis_output/region_entropy_dotplot.png / .pdf
        entropy_analysis_output/region_entropy_summary_with_range.csv
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Input / output setting
# =========================
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--residue-only", action="store_true", help="Save only the residue entropy panel under a separate filename.")
args = parser.parse_args()
output_dir = Path(__file__).resolve().parent / "entropy_analysis_output"
input_csv = output_dir / "residue_property_entropy_merged.csv"
os.makedirs(output_dir, exist_ok=True)

# same KLIFS region definition as entropy_analysis.ipynb
region_ranges = [
    ("beta-strand-I",   1, 3),
    ("glycine-rich loop", 4, 9),
    ("beta-strand-II",  10, 13),
    ("beta-strand-III", 14, 19),
    ("alpha-C helix",   20, 30),
    ("back loop",       31, 37),
    ("beta-strand-IV",  38, 41),
    ("beta-strand-V",   42, 44),
    ("gatekeeper",      45, 45),
    ("hinge region",    46, 48),
    ("linker",          49, 52),
    ("alpha-D helix",   53, 59),
    ("alpha-E helix",   60, 64),
    ("beta-strand-VI",  65, 67),
    ("catalytic loop",  68, 75),
    ("beta-strand-VII", 76, 78),
    ("beta-strand-VIII",79, 79),
    ("xDFG",            80, 83),
    ("activation loop", 84, 85),
]
region_order = [r[0] for r in region_ranges]

# (column, y-axis label) for each panel
panels = [
    ("Residue_entropy",       "Residue entropy (nats)"),
    ("Property_entropy_mean", "Mean property entropy (nats)"),
]

# plot style
POINT_COLOR = "#2a78d6"
MEAN_COLOR = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SHADE_COLOR = "#f3f2ef"
GROUP_WIDTH = 0.70        # horizontal span used by the points of one region
ANNOTATE_MIN_N = 3        # annotate min / max positions for regions with n >= this

# =========================
# 2. Load data
# =========================
df = pd.read_csv(input_csv)
df = df.sort_values("KLIFS_position").reset_index(drop=True)
assert len(df) == 85, f"Expected 85 KLIFS positions, got {len(df)}"

df["region_idx"] = df["Functional_region"].map({r: i for i, r in enumerate(region_order)})
assert df["region_idx"].notna().all(), "Unknown functional region in input"

# x coordinate: region index + offset by KLIFS order within region
def within_region_offsets(n):
    if n == 1:
        return np.array([0.0])
    return np.linspace(-GROUP_WIDTH / 2, GROUP_WIDTH / 2, n)

df["x"] = np.nan
for i, region in enumerate(region_order):
    mask = df["Functional_region"] == region
    df.loc[mask, "x"] = i + within_region_offsets(mask.sum())

# =========================
# 3. Region summary (n, mean, SD, range, extreme positions)
# =========================
summary_records = []
for region in region_order:
    sub = df[df["Functional_region"] == region]
    row = {
        "Functional_region": region,
        "n_positions": len(sub),
        "KLIFS_start": int(sub["KLIFS_position"].min()),
        "KLIFS_end": int(sub["KLIFS_position"].max()),
    }
    for col, _ in panels:
        vals = sub[col]
        row[f"{col}_mean"] = vals.mean()
        row[f"{col}_sd"] = vals.std(ddof=1) if len(sub) > 1 else np.nan
        row[f"{col}_min"] = vals.min()
        row[f"{col}_min_pos"] = int(sub.loc[vals.idxmin(), "KLIFS_position"])
        row[f"{col}_max"] = vals.max()
        row[f"{col}_max_pos"] = int(sub.loc[vals.idxmax(), "KLIFS_position"])
    summary_records.append(row)

summary_df = pd.DataFrame(summary_records)
summary_csv = os.path.join(output_dir, "region_entropy_summary_with_range.csv")
if not args.residue_only:
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved region summary: {summary_csv}")

# =========================
# 4. Plot
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": TEXT_SECONDARY,
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
})

plot_panels = panels[:1] if args.residue_only else panels
fig, axes = plt.subplots(len(plot_panels), 1, figsize=(13, 4.8) if args.residue_only else (13, 7.5), sharex=True, squeeze=False)
axes = axes[:, 0]

for ax, (col, ylabel) in zip(axes, plot_panels):
    # alternating region shading
    for i in range(len(region_order)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color=SHADE_COLOR, zorder=0, lw=0)

    # region mean
    for i, region in enumerate(region_order):
        n = int(summary_df.loc[i, "n_positions"])
        half = GROUP_WIDTH / 2 + 0.08 if n > 1 else 0.22
        ax.hlines(summary_df.loc[i, f"{col}_mean"], i - half, i + half,
                  color=MEAN_COLOR, lw=1.6, zorder=2)

    # individual positions (white edge ring separates close points)
    ax.scatter(df["x"], df[col], s=30, color=POINT_COLOR,
               edgecolor="white", linewidth=0.8, zorder=3)

    # annotate lowest / highest position of each region with its KLIFS number
    y_span = df[col].max() - df[col].min()
    for i, region in enumerate(region_order):
        if summary_df.loc[i, "n_positions"] < ANNOTATE_MIN_N:
            continue
        for kind, dy, va in (("min", -0.035, "top"), ("max", 0.035, "bottom")):
            pos = summary_df.loc[i, f"{col}_{kind}_pos"]
            point = df.loc[df["KLIFS_position"] == pos].iloc[0]
            ax.text(point["x"], point[col] + dy * y_span, str(pos),
                    ha="center", va=va, fontsize=6.5, color=TEXT_SECONDARY, zorder=4)

    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(region_order) - 0.4)
    y_min, y_max = df[col].min(), df[col].max()
    ax.set_ylim(max(0, y_min - 0.12 * y_span), y_max + 0.12 * y_span)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.6, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

x_labels = [f"{r} (n={n})" for r, n in zip(summary_df["Functional_region"], summary_df["n_positions"])]
axes[-1].set_xticks(range(len(region_order)))
axes[-1].set_xticklabels(x_labels, rotation=50, ha="right", rotation_mode="anchor")
axes[-1].set_xlabel("KLIFS functional region")

axes[0].set_title(("" if args.residue_only else "A   ") + "Residue-level entropy of KLIFS positions (482 human kinases)", loc="left", fontsize=10)
if not args.residue_only:
    axes[1].set_title("B   Property-level entropy of KLIFS positions (mean over 23 property bits)", loc="left", fontsize=10)

# legend
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([], [], marker="o", ls="", color=POINT_COLOR, markeredgecolor="white", markersize=6,
           label="KLIFS position"),
    Line2D([], [], color=MEAN_COLOR, lw=1.6, label="Region mean"),
]
axes[0].legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8, ncol=2,
               bbox_to_anchor=(1.0, 1.16))

plt.tight_layout()
output_stem = "region_entropy_dotplot_residue_only" if args.residue_only else "region_entropy_dotplot"
for ext in ("png", "pdf"):
    out_path = os.path.join(output_dir, f"{output_stem}.{ext}")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
plt.close()
