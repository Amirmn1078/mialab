import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    # todo: load the "results.csv" file from the mia-results directory
    csv_path = "C:/Users/amirm/Desktop/files/BME master/MIALab/mialab/mia-result/2025-12-14-17-24-40/results.csv"
    _sep_used = ';'  # <-- force semicolon separator since your file always uses ';'

    # output directory for plots
    out_dir = os.path.join(os.path.dirname(csv_path), 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load CSV (LONG format expected)
    # ------------------------------------------------------------------
    import csv, math

    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f, delimiter=_sep_used)
        rows = list(reader)

    headers = [h.strip().replace('\ufeff', '') for h in rows[0]]
    rows = rows[1:]
    headers_l = [h.lower() for h in headers]

    print(f"[debug] CSV headers: {headers}")

    def _idx(name):
        name = name.lower()
        for i, h in enumerate(headers_l):
            if name == h:
                return i
        return None

    label_idx = _idx('label')
    dice_idx = _idx('dice')
    jacc_idx = _idx('jacrd')
    prec_idx = _idx('prcison')
    reca_idx = _idx('snsvty')
    hd95_idx = _idx('hdrfdst')

    labels = ['WhiteMatter', 'GreyMatter', 'hippocampus', 'amygdala', 'thalamus']

    def _map_label(s):
        k = s.strip().lower().replace('_', '').replace(' ', '')
        if k == 'whitematter':
            return 'WhiteMatter'
        if k in ('greymatter', 'graymatter'):
            return 'GreyMatter'
        return s.strip().lower()

    def _to_float(x):
        try:
            v = float(x)
            return v if math.isfinite(v) else np.nan
        except Exception:
            return np.nan

    # ------------------------------------------------------------------
    # Generic plotting helper
    # ------------------------------------------------------------------
    def plot_metric(metric_idx, ylabel, fname):
        if metric_idx is None or label_idx is None:
            print(f"[skip] {ylabel}: column missing.")
            return

        per_label = {l: [] for l in labels}

        for r in rows:
            if metric_idx >= len(r) or label_idx >= len(r):
                continue
            lab = _map_label(r[label_idx])
            val = _to_float(r[metric_idx])
            if lab in per_label and not np.isnan(val):
                per_label[lab].append(val)

        data = [np.array(per_label[l]) for l in labels]
        if all(len(d) == 0 for d in data):
            print(f"[skip] {ylabel}: empty.")
            return

        plt.boxplot(data, tick_labels=labels)
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} per label')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.show()
        print(f"[ok] saved → {fname}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    plot_metric(dice_idx, 'Dice coefficient', 'dice_per_label.png')
    plot_metric(jacc_idx, 'Jaccard (IoU)', 'jaccard_per_label.png')
    plot_metric(prec_idx, 'Precision', 'precision_per_label.png')
    plot_metric(reca_idx, 'Recall / Sensitivity', 'recall_per_label.png')
    plot_metric(hd95_idx, 'Hausdorff Distance (HD95)', 'hd95_per_label.png')

    # ------------------------------------------------------------------
    # Save mid-slice of segmentation volumes
    # ------------------------------------------------------------------
    try:
        import SimpleITK as sitk

        seg_dir = os.path.dirname(csv_path)
        seg_out_dir = os.path.join(out_dir, "segmentation_slices")
        os.makedirs(seg_out_dir, exist_ok=True)

        seg_files = [f for f in os.listdir(seg_dir) if f.lower().endswith(".mha")]

        for fname in seg_files:
            img = sitk.ReadImage(os.path.join(seg_dir, fname))
            arr = sitk.GetArrayFromImage(img)

            z_mid = arr.shape[0] // 2
            sl = arr[z_mid]

            plt.imshow(sl, cmap="gray")
            plt.axis("off")
            out_png = os.path.join(seg_out_dir, fname.replace('.mha', '_mid.png'))
            plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"[seg] saved → {out_png}")

    except Exception as e:
        print(f"[seg error] {e}")

    # ------------------------------------------------------------------
    # Colored segmentation with legend
    # ------------------------------------------------------------------
    try:
        import SimpleITK as sitk
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch

        label_names = {
            1: "WhiteMatter",
            2: "GreyMatter",
            3: "hippocampus",
            4: "amygdala",
            5: "thalamus",
        }

        cmap = ListedColormap([
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
        ])
        norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

        seg_dir = os.path.dirname(csv_path)
        seg_out_dir = os.path.join(out_dir, "segmentation_colored")
        os.makedirs(seg_out_dir, exist_ok=True)

        for fname in seg_files:
            img = sitk.ReadImage(os.path.join(seg_dir, fname))
            arr = sitk.GetArrayFromImage(img)
            sl = arr[arr.shape[0] // 2]

            plt.imshow(sl, cmap=cmap, norm=norm)
            plt.axis("off")

            handles = [Patch(color=cmap(k), label=v) for k, v in label_names.items()]
            plt.legend(handles=handles, fontsize=7, framealpha=0.8)

            out_png = os.path.join(seg_out_dir, fname.replace('.mha', '_colored.png'))
            plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"[seg-vis] saved → {out_png}")

    except Exception as e:
        print(f"[seg-vis error] {e}")


if __name__ == '__main__':
    main()
