import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, random_split

from dataset_3D import LungNodule3DDataset
from train_3D_attention import LitAttentionModel

def get_best_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts: return None
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

if __name__ == "__main__":
    print("=== 🚀 Clinical utility and optimal threshold analysis ===")
    out_dir = "models/evaluation_results/clinical_value/"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Automatically load the strongest 3D Attention model
    ckpt_path = get_best_ckpt("models/checkpoints_attention/")
    print(f"📦 Load model checkpoint: {os.path.basename(ckpt_path)}")
    model = LitAttentionModel.load_from_checkpoint(ckpt_path).eval()

    # 2. Prepare the unaugmented validation set
    MANIFEST_PATH = "data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "data/processed/all_roi_cubes/"
    
    torch.manual_seed(42) # Keep the split identical to the previous evaluation
    full_dataset = LungNodule3DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=None)
    train_size = int(0.8 * len(full_dataset))
    _, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 3. Collect all prediction probabilities and ground-truth labels
    all_probs, all_labels = [], []
    print("⏳ Running inference on the full validation set...")
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 4. Calculate ROC curve values
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # 5. Core clinical statistic: Youden's J statistic
    # Youden's J = sensitivity + specificity - 1 = TPR - FPR
    # The maximum Youden's J identifies the threshold that best balances false negatives and false positives
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    opt_tpr = tpr[optimal_idx]  # Sensitivity (higher values mean fewer false negatives)
    opt_fpr = fpr[optimal_idx]  # 1 - specificity
    opt_spec = 1 - opt_fpr      # Specificity (higher values mean fewer false positives)

    # Print the key clinical metrics
    print("\n📊 === Key Clinical Decision Metrics ===")
    print(f"🔹 Overall diagnostic AUC   : {roc_auc:.4f}")
    print(f"⭐ Optimal operating threshold : {optimal_threshold:.4f} (default: 0.5000)")
    print(f"📈 Sensitivity  : {opt_tpr:.4f}  👉 false-negative rate: {1-opt_tpr:.4f}")
    print(f"📉 Specificity  : {opt_spec:.4f}  👉 false-positive rate: {opt_fpr:.4f}")
    print("===================================\n")

    # 6. Plot the clinical ROC curve with the optimal operating point
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Model ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Mark the optimal operating point with a red star
    plt.scatter(opt_fpr, opt_tpr, color='red', s=120, zorder=5, marker='*', 
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    # Add metric annotations
    plt.annotate(f'Sensitivity: {opt_tpr:.2f}\nSpecificity: {opt_spec:.2f}', 
                 (opt_fpr, opt_tpr), textcoords="offset points", xytext=(15,-15), 
                 ha='left', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.1))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Clinical ROC Curve & Optimal Operating Point', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(out_dir, "clinical_roc_optimal_point.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Clinical operating-point ROC curve saved to: {plot_path}")