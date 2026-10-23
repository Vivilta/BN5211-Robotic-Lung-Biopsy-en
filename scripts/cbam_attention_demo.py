
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_3D_attention import LitAttentionModel

DATA_DIR = ROOT / "data" / "demo_nodules"
OUTPUT_DIR = ROOT / "ppt_results" / "cbam_attention"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT = (
    ROOT / "models/checkpoints_attention"
    / "3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt"
)


class AttentionExtractor:
    def __init__(self, model):
        self.channel = {}
        self.spatial = {}
        self.handles = []

        blocks = {
            "Layer 1": model.model.layer1[-1],
            "Layer 2": model.model.layer2[-1],
        }

        for name, block in blocks.items():
            self.handles.append(
                block.ca.register_forward_hook(
                    self.make_hook(self.channel, name)
                )
            )
            self.handles.append(
                block.sa.register_forward_hook(
                    self.make_hook(self.spatial, name)
                )
            )

    @staticmethod
    def make_hook(store, name):
        def hook(module, inputs, output):
            store[name] = output.detach().cpu().clone()
        return hook

    def clear(self):
        self.channel.clear()
        self.spatial.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def get_views(volume):
    z, y, x = [s // 2 for s in volume.shape]
    return [
        volume[z, :, :],
        volume[:, y, :],
        volume[:, :, x],
    ]


def normalize_image(image):
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1)


def save_spatial(case_name, roi, spatial_maps, probability):
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))

    titles = [
        "Original ROI",
        "Layer 1 Spatial Attention",
        "Layer 2 Spatial Attention",
        "Layer 2 Overlay"
    ]

    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11)

    original_views = get_views(roi)
    layer1_views = get_views(spatial_maps["Layer 1"])
    layer2_views = get_views(spatial_maps["Layer 2"])

    for row in range(3):
        original = normalize_image(original_views[row])
        att1 = layer1_views[row]
        att2 = layer2_views[row]

        axes[row, 0].imshow(
            original, cmap="gray", interpolation="nearest"
        )

        axes[row, 1].imshow(
            att1, cmap="viridis", vmin=0, vmax=1
        )

        axes[row, 2].imshow(
            att2, cmap="viridis", vmin=0, vmax=1
        )

        axes[row, 3].imshow(
            original, cmap="gray", interpolation="nearest"
        )
        axes[row, 3].imshow(
            att2, cmap="viridis", vmin=0, vmax=1, alpha=0.45
        )

        axes[row, 0].set_ylabel(f"Axis {row}")

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"{case_name}\n"
        f"Positive-class probability: {probability:.2%}",
        fontsize=14
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output = OUTPUT_DIR / f"{case_name}_spatial_attention.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)


def save_channel(case_name, channel_weights):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    for ax, (name, weights) in zip(axes, channel_weights.items()):
        values = weights.reshape(-1)

        ax.bar(np.arange(len(values)), values)
        ax.set_title(f"{name} Channel Attention")
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Attention Weight")
        ax.set_ylim(0, 1)

    fig.suptitle(case_name)
    fig.tight_layout()

    output = OUTPUT_DIR / f"{case_name}_channel_attention.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)


def main():
    model = LitAttentionModel.load_from_checkpoint(
        str(CKPT),
        map_location="cpu"
    )
    model.eval()

    extractor = AttentionExtractor(model)

    try:
        files = sorted(DATA_DIR.glob("*.npy"))

        if not files:
            raise FileNotFoundError("No ROI files found")

        for file in files:
            extractor.clear()

            roi = np.load(file, allow_pickle=False)

            if roi.shape != (32, 32, 32):
                raise ValueError(f"Invalid ROI: {file.name}")

            tensor = (
                torch.from_numpy(roi)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )

            with torch.no_grad():
                logits = model(tensor)
                probability = torch.softmax(
                    logits, dim=1
                )[0, 1].item()

            spatial_maps = {}
            channel_weights = {}

            for name in ["Layer 1", "Layer 2"]:
                spatial = extractor.spatial[name]

                # 上采样至原始 ROI 尺寸
                spatial_up = F.interpolate(
                    spatial,
                    size=(32, 32, 32),
                    mode="trilinear",
                    align_corners=False
                )

                spatial_maps[name] = (
                    spatial_up[0, 0].numpy()
                )

                channel_weights[name] = (
                    extractor.channel[name][0, :, 0, 0, 0]
                    .numpy()
                )

                print(
                    f"{file.stem} / {name}: "
                    f"Spatial mean={spatial_maps[name].mean():.4f}, "
                    f"Channel mean={channel_weights[name].mean():.4f}"
                )

            save_spatial(
                file.stem,
                roi,
                spatial_maps,
                probability
            )

            save_channel(
                file.stem,
                channel_weights
            )

    finally:
        extractor.close()


if __name__ == "__main__":
    main()
