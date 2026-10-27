
import sys
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_3D_attention import LitAttentionModel
from cbam_attention_demo import AttentionExtractor

DATA_DIR = ROOT / "data" / "demo_nodules"
OUTPUT_DIR = ROOT / "ppt_results" / "attention_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT = (
    ROOT / "models/checkpoints_attention"
    / "3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt"
)

LAYERS = ["Layer 1", "Layer 2"]
THRESHOLD = 0.8


def compute_statistics(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "high_weight_fraction": float(np.mean(values > THRESHOLD)),
    }


def correlation_matrix(vectors):
    """
    vectors: shape (number_of_cases, number_of_channels)
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    n = vectors.shape[0]

    corr = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            x = vectors[i]
            y = vectors[j]

            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                continue

            corr[i, j] = np.corrcoef(x, y)[0, 1]

    return corr


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


def plot_correlation(corr, case_names, layer):
    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(case_names)))
    ax.set_yticks(range(len(case_names)))

    ax.set_xticklabels(case_names)
    ax.set_yticklabels(case_names)

    ax.set_title(f"{layer} Channel Attention Correlation")

    for i in range(len(case_names)):
        for j in range(len(case_names)):
            value = corr[i, j]

            label = "N/A" if np.isnan(value) else f"{value:.3f}"

            ax.text(
                j, i, label,
                ha="center",
                va="center",
                color="black",
                fontsize=11
            )

    fig.colorbar(image, ax=ax, label="Pearson r")
    fig.tight_layout()

    filename = (
        "layer1_channel_correlation.png"
        if layer == "Layer 1"
        else "layer2_channel_correlation.png"
    )

    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print("Saved:", path)


def plot_spatial_statistics(rows):
    case_names = sorted(set(row["case"] for row in rows))
    x = np.arange(len(case_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for index, layer in enumerate(LAYERS):
        values = [
            next(
                row["high_weight_fraction"]
                for row in rows
                if row["case"] == case
                and row["layer"] == layer
                and row["attention_type"] == "spatial"
            )
            for case in case_names
        ]

        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=layer
        )

    ax.set_xticks(x)
    ax.set_xticklabels(case_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of weights > 0.8")
    ax.set_title("Spatial Attention: High-weight Fraction")
    ax.legend()

    fig.tight_layout()

    path = OUTPUT_DIR / "spatial_high_weight_fraction.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print("Saved:", path)


def main():
    model = LitAttentionModel.load_from_checkpoint(
        str(CKPT),
        map_location="cpu"
    )
    model.eval()

    extractor = AttentionExtractor(model)

    files = sorted(DATA_DIR.glob("*.npy"))

    if not files:
        raise FileNotFoundError(f"No .npy files in {DATA_DIR}")

    statistics = []
    channel_vectors = {layer: [] for layer in LAYERS}
    case_names = []

    try:
        for file in files:
            roi = np.load(file, allow_pickle=False)

            if roi.shape != (32, 32, 32):
                raise ValueError(f"Invalid ROI shape: {file.name}")

            if not np.isfinite(roi).all():
                raise ValueError(f"Non-finite values: {file.name}")

            tensor = (
                torch.from_numpy(roi)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )

            extractor.clear()

            with torch.no_grad():
                logits = model(tensor)
                probability = torch.softmax(
                    logits, dim=1
                )[0, 1].item()

            case_names.append(file.stem)

            print(f"\nCase: {file.stem}")
            print(f"Positive-class probability: {probability:.2%}")

            for layer in LAYERS:
                channel = (
                    extractor.channel[layer]
                    .numpy()
                    .reshape(-1)
                )

                # Use the original spatial attention weights without upsampling
                spatial = (
                    extractor.spatial[layer]
                    .numpy()
                    .reshape(-1)
                )

                channel_vectors[layer].append(channel)

                for attention_type, values in [
                    ("channel", channel),
                    ("spatial", spatial),
                ]:
                    stats = compute_statistics(values)

                    row = {
                        "case": file.stem,
                        "layer": layer,
                        "attention_type": attention_type,
                        "positive_class_probability": probability,
                        **stats,
                    }

                    statistics.append(row)

                    print(
                        f"  {layer} / {attention_type}: "
                        f"mean={stats['mean']:.4f}, "
                        f"std={stats['std']:.4f}, "
                        f"fraction>0.8="
                        f"{stats['high_weight_fraction']:.2%}"
                    )

    finally:
        extractor.close()

    fieldnames = [
        "case",
        "layer",
        "attention_type",
        "positive_class_probability",
        "mean",
        "std",
        "min",
        "max",
        "high_weight_fraction",
    ]

    save_csv(
        OUTPUT_DIR / "attention_statistics.csv",
        statistics,
        fieldnames
    )

    correlation_rows = []

    for layer in LAYERS:
        vectors = np.stack(channel_vectors[layer])
        corr = correlation_matrix(vectors)

        plot_correlation(corr, case_names, layer)

        for i, case_a in enumerate(case_names):
            for j, case_b in enumerate(case_names):
                correlation_rows.append({
                    "layer": layer,
                    "case_a": case_a,
                    "case_b": case_b,
                    "pearson_r": corr[i, j],
                })

    save_csv(
        OUTPUT_DIR / "channel_correlations.csv",
        correlation_rows,
        ["layer", "case_a", "case_b", "pearson_r"]
    )

    plot_spatial_statistics(statistics)

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
