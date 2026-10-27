
import sys
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Project root directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_3D_attention import LitAttentionModel
from train_3D import LitLungNoduleModel

# File paths
DATA_DIR = ROOT / "data" / "registration_candidates" / "rois"
CKPT_DIR = ROOT / "models" / "checkpoints_attention"
OUTPUT_DIR = ROOT / "ppt_results" / "registered_inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "3D ResNet18": CKPT_DIR / "3d_resnet18-epoch=27-val_acc=0.81.ckpt",
    "3D CBAM": CKPT_DIR / "3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt",
}

MODEL_CLASSES = {
    "3D ResNet18": LitLungNoduleModel,
    "3D CBAM": LitAttentionModel,
}

def load_model(model_name):
    ckpt_path = CHECKPOINTS[model_name]

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    print(f"\nLoading {model_name}")
    print(f"Checkpoint: {ckpt_path.name}")

    model = MODEL_CLASSES[model_name].load_from_checkpoint(
        str(ckpt_path),
        map_location="cpu"
    )

    model.eval()
    return model


def predict(model, npy_path):
    roi = np.load(npy_path, allow_pickle=False)

    if roi.shape != (32, 32, 32):
        raise ValueError(f"Invalid ROI shape: {roi.shape}")

    if not np.isfinite(roi).all():
        raise ValueError(f"Non-finite values in {npy_path.name}")

    # (D,H,W) -> (B,C,D,H,W)
    tensor = (
        torch.from_numpy(roi)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    with torch.inference_mode():
        logits = model(tensor)
        if logits.shape != (1, 2):
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

        probabilities = torch.softmax(logits, dim=1)

    return float(probabilities[0, 1].item())


def main():
    files = sorted(DATA_DIR.glob("*.npy"))

    if not files:
        raise FileNotFoundError("No .npy files found")

    models = {
        name: load_model(name)
        for name in CHECKPOINTS
    }

    results = []

    for file in files:
        print(f"\nCase: {file.name}")

        for model_name, model in models.items():
            probability = predict(model, file)

            print(f"  {model_name}: {probability:.2%}")

            results.append({
                "case": file.stem,
                "model": model_name,
                "positive_class_probability": probability
            })

    csv_path = OUTPUT_DIR / "predictions.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "model",
                "positive_class_probability"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    # Plot model comparison
    case_names = [f.stem for f in files]
    x = np.arange(len(case_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    for index, model_name in enumerate(CHECKPOINTS):
        values = [
            next(
                r["positive_class_probability"]
                for r in results
                if r["case"] == case
                and r["model"] == model_name
            )
            for case in case_names
        ]

        positions = x + (index - 0.5) * width

        bars = ax.bar(
            positions,
            values,
            width,
            label=model_name
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=3
        )

    ax.set_xticks(x)
    ax.set_xticklabels(case_names, rotation=15, ha="right")
    ax.set_ylabel("Positive-class probability")
    ax.set_ylim(0, 1.1)
    ax.set_title("3D Model Predictions on Three Lung Nodule ROIs")
    ax.legend()

    fig.tight_layout()

    image_path = OUTPUT_DIR / "model_comparison.png"
    fig.savefig(image_path, dpi=300)
    plt.close(fig)

    print("\nSaved results:")
    print(csv_path)
    print(image_path)


if __name__ == "__main__":
    main()
