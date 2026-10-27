import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, random_split

# Import the 3D dataset and model wrappers
from dataset_3D import LungNodule3DDataset
from train_3D import LitLungNoduleModel          # 3D ResNet
from train_3D_vnet import LitLungNoduleVNet3D # 3D VNet

def get_best_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

def evaluate_model(model, val_loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def print_metrics(name, y_true, y_pred, y_probs):
    print(f"\n--- {name} Evaluation Metrics ---")
    print(f"💡 Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"🎯 Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"🔍 Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"⚖️ F1-Score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"📈 AUC-ROC       : {roc_auc_score(y_true, y_probs):.4f}")

if __name__ == "__main__":
    print("=== 🚀 Starting the 3D model evaluation pipeline ===")
    out_dir = "models/evaluation_results/"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Prepare an unaugmented validation set for fair evaluation
    torch.manual_seed(42)
    full_dataset = LungNodule3DDataset("data/processed/final_dataset_manifest.csv", "data/processed/all_roi_cubes/", transform=None)
    train_size = int(0.8 * len(full_dataset))
    _, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    results = {}
    
    # 2. Evaluate 3D ResNet
    ckpt_resnet = get_best_ckpt("models/checkpoints/")
    if ckpt_resnet:
        print(f"\n📦 Load the best 3D ResNet checkpoint: {os.path.basename(ckpt_resnet)}")
        model_resnet = LitLungNoduleModel.load_from_checkpoint(ckpt_resnet)
        y_true, y_pred, y_probs = evaluate_model(model_resnet, val_loader)
        results['ResNet3D'] = (y_true, y_pred, y_probs)
        print_metrics('3D ResNet18', y_true, y_pred, y_probs)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Reds', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix (3D ResNet18)')
        plt.savefig(os.path.join(out_dir, '3d_resnet_cm.png'), dpi=300)
        plt.close()

    # 3. Evaluate 3D VNet
    ckpt_vnet = get_best_ckpt("models/checkpoints_3d_vnet/")
    if ckpt_vnet:
        print(f"\n📦 Load the best 3D VNet checkpoint: {os.path.basename(ckpt_vnet)}")
        model_vnet = LitLungNoduleVNet3D.load_from_checkpoint(ckpt_vnet)
        y_true, y_pred, y_probs = evaluate_model(model_vnet, val_loader)
        results['VNet3D'] = (y_true, y_pred, y_probs)
        print_metrics('3D VNet', y_true, y_pred, y_probs)
        
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Greens', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix (3D VNet)')
        plt.savefig(os.path.join(out_dir, '3d_vnet_cm.png'), dpi=300)
        plt.close()

    # 4. Plot the combined ROC comparison curve
    if len(results) == 2:
        plt.figure(figsize=(7, 6))
        colors = {'ResNet3D': 'deeppink', 'VNet3D': 'forestgreen'}
        for name, (y_t, y_p, y_prob) in results.items():
            fpr, tpr, _ = roc_curve(y_t, y_prob)
            auc = roc_auc_score(y_t, y_prob)
            plt.plot(fpr, tpr, color=colors[name], lw=2, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('3D Models ROC Comparison (Full Volumetric)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '3d_combined_roc.png'), dpi=300)
        plt.close()
        print(f"\n✅ Combined 3D ROC comparison curve generated!")