import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION (UNET)
# =========================

files = {
    "UNet – Base (no preprocessing)":
        r"C:/Users/amirm/Desktop/files/BME master/MIALab/mialab/mia-result/baseUnet/resultsbase2Unet.csv",

    "UNet – Group A (Foreground Z-score + Clipping)":
        r"C:/Users/amirm/Desktop/files/BME master/MIALab/mialab/mia-result/GroupA-(Intensity Normalization & Standardization)-Foreground-only Z-Score Normalization&Robust Intensity ClippingUnet/resultsGroupA-(Intensity Normalization & Standardization)-F.csv",

    "UNet – Group B (N4 Bias Field Correction)":
        r"C:/Users/amirm/Desktop/files/BME master/MIALab/mialab/mia-result/GroupB-(Inhomogeneity & Artifact Correction)N4 Bias Field CorrectionUnet/resultsGroupB-(Inhomogeneity & Artifact Correction)N4 Bias Field CorrectionUnet.csv",

    "UNet – Group A + Group B":
        r"C:/Users/amirm/Desktop/files/BME master/MIALab/mialab/mia-result/GroupA+GroupBUnet/resultsGroupA+GroupBUnet.csv",
}

# Index of metrics inside the semicolon-separated string
METRIC_INDEX = {
    "Dice": 2,
    "Hausdorff": 3,
    "Recall": 4,
    "Sensitivity": 5
}

# =========================
# PROCESSING
# =========================

rows = []

for name, path in files.items():
    df = pd.read_csv(path)

    metrics = {k: [] for k in METRIC_INDEX.keys()}

    for cell in df.iloc[:, 0]:
        if isinstance(cell, str) and ";" in cell:
            parts = cell.split(";")
            try:
                for metric, idx in METRIC_INDEX.items():
                    metrics[metric].append(float(parts[idx]))
            except (ValueError, IndexError):
                continue

    row = {"Pipeline": name}
    for metric, values in metrics.items():
        if len(values) > 0:
            row[metric] = f"{np.mean(values):.3f} ± {np.std(values):.3f}"
        else:
            row[metric] = "N/A"

    rows.append(row)

summary_table = pd.DataFrame(rows)

# =========================
# OUTPUTS
# =========================

# Print
print(summary_table.to_string(index=False))

# Save CSV
summary_table.to_csv("unet_preprocessing_summary_table.csv", index=False)

# Save IMAGE
fig, ax = plt.subplots(figsize=(14, 3))
ax.axis("off")

table = ax.table(
    cellText=summary_table.values,
    colLabels=summary_table.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.7)

plt.tight_layout()
plt.savefig("unet_preprocessing_summary_table.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved:")
print(" - unet_preprocessing_summary_table.csv")
print(" - unet_preprocessing_summary_table.png")
