import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

input_dir = Path("data/demo_nodules")
output_dir = Path("ppt_results/roi")
output_dir.mkdir(parents=True, exist_ok=True)

for file in sorted(input_dir.glob("*.npy")):
    volume = np.load(file, allow_pickle=False)

    if volume.shape != (32, 32, 32):
        print(f"Skip {file.name}: unexpected shape {volume.shape}")
        continue

    z, y, x = [s // 2 for s in volume.shape]

    views = [
        ("Axis 0", volume[z, :, :]),
        ("Axis 1", volume[:, y, :]),
        ("Axis 2", volume[:, :, x]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    for ax, (title, image) in zip(axes, views):
        ax.imshow(image, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(file.stem, fontsize=12)
    plt.tight_layout()

    save_path = output_dir / f"{file.stem}_views.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", save_path)