
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
OUTPUT_DIR = ROOT / "ppt_results" / "gradcam_layers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT = (
    ROOT / "models/checkpoints_attention"
    / "3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt"
)

TARGET_CLASS = 1


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
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, target_class=1):
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        logits = self.model(tensor)

        if logits.shape != (1, 2):
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors")

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
            size=tensor.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()

        raw_max = float(cam.max())
        raw_mean = float(cam.mean())

        # 只用于每张热力图内部的可视化归一化
        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, raw_max, raw_mean

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def normalize_image(image):
    low, high = np.percentile(image, [1, 99])

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    return np.clip((image - low) / (high - low), 0, 1)


def get_views(volume):
    z, y, x = [s // 2 for s in volume.shape]

    return [
        volume[z, :, :],
        volume[:, y, :],
        volume[:, :, x]
    ]


def save_comparison(case_name, roi, cams, probability):
    original_views = get_views(roi)

    layer_names = list(cams.keys())
    cam_views = {
        name: get_views(cam)
        for name, cam in cams.items()
    }

    fig, axes = plt.subplots(
        3, 5,
        figsize=(16, 10)
    )

    titles = [
        "Original CT",
        "Layer 1 CBAM\n16 x 16 x 16",
        "Layer 2 Conv\n8 x 8 x 8",
        "Layer 2 CBAM\n8 x 8 x 8",
        "Final CBAM Overlay"
    ]

    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=12)

    for row in range(3):
        image = normalize_image(original_views[row])

        axes[row, 0].imshow(
            image,
            cmap="gray",
            interpolation="nearest"
        )

        for col, layer_name in enumerate(layer_names, start=1):
            heatmap = cam_views[layer_name][row]

            axes[row, col].imshow(
                heatmap,
                cmap="jet",
                vmin=0,
                vmax=1,
                interpolation="bilinear"
            )

        axes[row, 4].imshow(
            image,
            cmap="gray",
            interpolation="nearest"
        )

        axes[row, 4].imshow(
            cam_views["Layer 2 CBAM"][row],
            cmap="jet",
            vmin=0,
            vmax=1,
            alpha=0.4,
            interpolation="bilinear"
        )

        axes[row, 0].set_ylabel(
            f"Axis {row}",
            fontsize=12
        )

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"{case_name}\n"
        f"Class 1 Probability: {probability:.2%}",
        fontsize=16
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = OUTPUT_DIR / f"{case_name}_layer_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", path)


def main():
    model = LitAttentionModel.load_from_checkpoint(
        str(CKPT),
        map_location="cpu"
    )
    model.eval()

    target_layers = {
        "Layer 1 CBAM": model.model.layer1[-1],
        "Layer 2 Conv": model.model.layer2[0],
        "Layer 2 CBAM": model.model.layer2[-1],
    }

    files = sorted(DATA_DIR.glob("*.npy"))

    if not files:
        raise FileNotFoundError(f"No ROI files in {DATA_DIR}")

    for file in files:
        roi = np.load(file, allow_pickle=False)

        if roi.shape != (32, 32, 32):
            raise ValueError(f"Invalid ROI shape: {file.name}")

        if not np.isfinite(roi).all():
            raise ValueError(f"Non-finite ROI values: {file.name}")

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
            )[0, TARGET_CLASS].item()

        cams = {}

        print(f"\nCase: {file.name}")
        print(f"Class 1 probability: {probability:.2%}")

        # 每次只注册一个目标层，避免 Hook 相互影响
        for name, layer in target_layers.items():
            gradcam = GradCAM3D(model, layer)

            try:
                cam, raw_max, raw_mean = gradcam.generate(
                    tensor,
                    target_class=TARGET_CLASS
                )
                cams[name] = cam

                print(
                    f"  {name}: "
                    f"raw_max={raw_max:.6f}, "
                    f"raw_mean={raw_mean:.6f}"
                )
            finally:
                gradcam.close()

        save_comparison(
            file.stem,
            roi,
            cams,
            probability
        )


if __name__ == "__main__":
    main()
