
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
OUTPUT_DIR = ROOT / "ppt_results" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT = (
    ROOT / "models" / "checkpoints_attention"
    / "3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt"
)


class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, inputs, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, target_class=1):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        score = logits[0, target_class]
        score.backward()

        weights = self.gradients.mean(
            dim=(2, 3, 4),
            keepdim=True
        )

        cam = (weights * self.activations).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=x.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def normalize_for_display(image):
    image = np.asarray(image, dtype=np.float32)
    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high <= low:
        return np.zeros_like(image)

    return np.clip((image - low) / (high - low), 0, 1)


def save_visualization(case_name, roi, cam):
    z, y, x = [s // 2 for s in roi.shape]

    views = [
        ("Axis 0", roi[z, :, :], cam[z, :, :]),
        ("Axis 1", roi[:, y, :], cam[:, y, :]),
        ("Axis 2", roi[:, :, x], cam[:, :, x]),
    ]

    fig, axes = plt.subplots(
        3, 3,
        figsize=(10, 10)
    )

    column_titles = [
        "Original ROI",
        "Grad-CAM",
        "Overlay"
    ]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=13)

    for row, (axis_name, image, heatmap) in enumerate(views):
        display_image = normalize_for_display(image)

        axes[row, 0].imshow(
            display_image,
            cmap="gray",
            interpolation="nearest"
        )

        axes[row, 1].imshow(
            heatmap,
            cmap="jet",
            vmin=0,
            vmax=1,
            interpolation="bilinear"
        )

        axes[row, 2].imshow(
            display_image,
            cmap="gray",
            interpolation="nearest"
        )

        axes[row, 2].imshow(
            heatmap,
            cmap="jet",
            vmin=0,
            vmax=1,
            alpha=0.4,
            interpolation="bilinear"
        )

        axes[row, 0].set_ylabel(
            axis_name,
            fontsize=12
        )

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"3D Grad-CAM: {case_name}\nTarget class: 1",
        fontsize=15
    )

    fig.tight_layout()

    save_path = OUTPUT_DIR / f"{case_name}_gradcam.png"

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", save_path)


def main():
    print("Loading CBAM checkpoint...")

    model = LitAttentionModel.load_from_checkpoint(
        str(CKPT),
        map_location="cpu"
    )

    model.eval()

    target_layer = model.model.layer2[-1]
    gradcam = GradCAM3D(model, target_layer)

    try:
        files = sorted(DATA_DIR.glob("*.npy"))

        if not files:
            raise FileNotFoundError(
                f"No .npy files found in {DATA_DIR}"
            )

        for file in files:
            print("\nProcessing:", file.name)

            roi = np.load(file, allow_pickle=False)

            if roi.shape != (32, 32, 32):
                raise ValueError(
                    f"Unexpected ROI shape: {roi.shape}"
                )

            if not np.isfinite(roi).all():
                raise ValueError(
                    f"Non-finite ROI values: {file.name}"
                )

            tensor = (
                torch.from_numpy(roi)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )

            # Grad-CAM requires gradients, so inference_mode() cannot be used.
            cam = gradcam.generate(
                tensor,
                target_class=1
            )

            save_visualization(
                file.stem,
                roi,
                cam
            )

    finally:
        gradcam.close()


if __name__ == "__main__":
    main()
