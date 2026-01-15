import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter

# ----------------- CONFIG -----------------
CSV_BASE = r"C:\Users\amirm\Desktop\files\BME master\MIALab\mialab\mia-result\baseUnet\resultsbase2Unet.csv"
CSV_BEST = r"C:\Users\amirm\Desktop\files\BME master\MIALab\mialab\mia-result\GroupA+GroupBUnet\resultsGroupA+GroupBUnet.csv"

REGIONS = ["Amygdala", "GreyMatter", "Hippocampus", "Thalamus", "WhiteMatter"]
# ------------------------------------------


def load_results(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    header = lines[0].split(";")
    rows = [ln.split(";") for ln in lines[1:]]
    df = pd.DataFrame(rows, columns=header)
    df.columns = [c.strip() for c in df.columns]
    df["LABEL"] = df["LABEL"].astype(str).str.strip()
    df["DICE"] = pd.to_numeric(df["DICE"], errors="coerce")
    return df


df_base = load_results(CSV_BASE)
df_best = load_results(CSV_BEST)

base_data = [df_base.loc[df_base["LABEL"] == r, "DICE"].dropna().to_numpy() for r in REGIONS]
best_data = [df_best.loc[df_best["LABEL"] == r, "DICE"].dropna().to_numpy() for r in REGIONS]

x = np.arange(len(REGIONS))
offset = 0.20

# --- monochromatic blue palette ---
BASE_COLOR = "#C6DBEF"   # very light blue
BEST_COLOR = "#2171B5"   # strong blue
EDGE_COLOR = "#1B2A41"   # dark blue-gray

fig, ax = plt.subplots(figsize=(12, 6))

common_kwargs = dict(
    widths=0.32,
    patch_artist=True,
    manage_ticks=False,
    showfliers=True,
    boxprops=dict(linewidth=1.2, edgecolor=EDGE_COLOR),
    whiskerprops=dict(linewidth=1.2, color=EDGE_COLOR),
    capprops=dict(linewidth=1.2, color=EDGE_COLOR),
    medianprops=dict(linewidth=2.0, color=EDGE_COLOR),
    flierprops=dict(marker='o', markersize=4,
                    markerfacecolor="white",
                    markeredgecolor=EDGE_COLOR,
                    alpha=0.8)
)

bp_base = ax.boxplot(base_data, positions=x - offset, **common_kwargs)
bp_best = ax.boxplot(best_data, positions=x + offset, **common_kwargs)

# Apply colors
for b in bp_base["boxes"]:
    b.set_facecolor(BASE_COLOR)
for b in bp_best["boxes"]:
    b.set_facecolor(BEST_COLOR)

# Force correct x-axis labels
ax.xaxis.set_major_locator(FixedLocator(x))
ax.xaxis.set_major_formatter(FixedFormatter(REGIONS))
ax.set_xlim(-0.6, len(REGIONS) - 0.4)

# Labels & title
ax.set_ylabel("Dice coefficient", fontsize=12)
ax.set_title("Dice score per brain region (U-Net: Base vs Best preprocessing))", fontsize=14, pad=12)
ax.set_ylim(0.25, 0.90)

# Subtle grid
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
ax.set_axisbelow(True)

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
ax.legend(
    handles=[
        Patch(facecolor=BASE_COLOR, edgecolor=EDGE_COLOR, label="Base preprocessing"),
        Patch(facecolor=BEST_COLOR, edgecolor=EDGE_COLOR, label="Best preprocessing"),
    ],
    frameon=False,
    loc="lower right"
)

plt.tight_layout()
plt.show()
