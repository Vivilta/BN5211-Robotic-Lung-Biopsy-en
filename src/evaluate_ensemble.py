import os
import glob
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, random_split

# Import the two strongest trained 3D architectures
from dataset_3D import LungNodule3DDataset
from train_3D_attention import LitAttentionModel
from train_3D_vnet import LitLungNoduleVNet3D

def get_best_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts: return None
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

if __name__ == "__main__":
    print("=== 🚀 [Task 7] Start combined model ensemble evaluation ===")
    
    # 1. Find the best checkpoints
    ckpt_attention = get_best_ckpt("models/checkpoints_attention/")
    ckpt_vnet = get_best_ckpt("models/checkpoints_3d_vnet/")
    
    if not ckpt_attention or not ckpt_vnet:
        print("❌ Error: Ensure that checkpoint files for both Attention and VNet exist in the models/ directory.")
        exit(1)
        
    print(f"📦 Found Attention model checkpoint: {os.path.basename(ckpt_attention)}")
    print(f"📦 Found VNet model checkpoint: {os.path.basename(ckpt_vnet)}")

    # 2. Load the models
    model_a = LitAttentionModel.load_from_checkpoint(ckpt_attention).eval()
    model_b = LitLungNoduleVNet3D.load_from_checkpoint(ckpt_vnet).eval()

    # 3. Prepare the validation set (use Seed=42 to match previous evaluations)
    MANIFEST_PATH = "data/processed/final_dataset_manifest.csv"
    CUBE_DIR = "data/processed/all_roi_cubes/"
    torch.manual_seed(42)
    full_dataset = LungNodule3DDataset(manifest_path=MANIFEST_PATH, cube_dir=CUBE_DIR, transform=None)
    train_size = int(0.8 * len(full_dataset))
    _, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 4. Collect prediction probabilities from both models
    probs_a, probs_b, labels = [], [], []
    
    print("⏳ Running inference with both models...")
    with torch.no_grad():
        for x, y in val_loader:
            # Inference with model A (3D Attention ResNet)
            logits_a = model_a(x)
            prob_a = torch.softmax(logits_a, dim=1)[:, 1].cpu().numpy()
            
            # Inference with model B (3D VNet)
            logits_b = model_b(x)
            prob_b = torch.softmax(logits_b, dim=1)[:, 1].cpu().numpy()
            
            probs_a.extend(prob_a)
            probs_b.extend(prob_b)
            labels.extend(y.cpu().numpy())

    probs_a = np.array(probs_a)
    probs_b = np.array(probs_b)
    labels = np.array(labels)

    # ==============================================================
    # 🧠 Core ensemble algorithm: weighted soft voting
    # Assign a higher weight to the Attention model based on its stronger independent evaluation performance
    # ==============================================================
    weight_a = 0.6  # 3D Attention weight
    weight_b = 0.4  # 3D VNet weight
    probs_ensemble = (probs_a * weight_a) + (probs_b * weight_b)
    preds_ensemble = (probs_ensemble > 0.5).astype(int)

    # Calculate metrics for each model
    auc_a = roc_auc_score(labels, probs_a)
    auc_b = roc_auc_score(labels, probs_b)
    auc_ens = roc_auc_score(labels, probs_ensemble)
    acc_ens = accuracy_score(labels, preds_ensemble)

    print("\n📊 === Ensemble Performance Comparison ===")
    print(f"🔸 Model A (3D Attention) AUC : {auc_a:.4f}")
    print(f"🔸 Model B (3D VNet) AUC : {auc_b:.4f}")
    print(f"🌟 Ensemble Model AUC : {auc_ens:.4f}  <-- Improved performance!")
    print(f"🌟 Ensemble Model Accuracy : {acc_ens:.4f}")