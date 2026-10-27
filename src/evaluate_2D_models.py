import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from torch.utils.data import DataLoader, random_split

# Import the 2D dataset and model wrappers
from dataset_2D import LungNodule2DDataset
from train_2D import LitLungNoduleModel2D       # 2D ResNet
from train_2D_densenet import LitLungNoduleDenseNet2D # 2D DenseNet

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
    print("=== 🚀 Starting the 2D model evaluation pipeline ===")
    out_dir = "models/evaluation_results/"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Prepare an unaugmented validation set for fair evaluation
    torch.manual_seed(42)
    full_dataset = LungNodule2DDataset("data/processed/final_dataset_manifest.csv", "data/processed/all_roi_cubes/", transform=None)
    train_size = int(0.8 * len(full_dataset))
    _, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    results = {}
    
    # 2. Evaluate 2D ResNet
    ckpt_resnet = get_best_ckpt("models/checkpoints_2d/")
    if ckpt_resnet:
        print(f"\n📦 Load the best 2D ResNet checkpoint: {os.path.basename(ckpt_resnet)}")
        model_resnet = LitLungNoduleModel2D.load_from_checkpoint(ckpt_resnet)
        y_true, y_pred, y_probs = evaluate_model(model_resnet, val_loader)
        results['ResNet18'] = (y_true, y_pred, y_probs)
        print_metrics('2D ResNet18', y_true, y_pred, y_probs)
        
        # Plot the individual confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Oranges', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix (2D ResNet18)')
        plt.savefig(os.path.join(out_dir, '2d_resnet_cm.png'), dpi=300)
        plt.close()

    # 3. Evaluate 2D DenseNet
    ckpt_densenet = get_best_ckpt("models/checkpoints_2d_densenet/")
    if ckpt_densenet:
        print(f"\n📦 Load the best 2D DenseNet checkpoint: {os.path.basename(ckpt_densenet)}")
        model_densenet = LitLungNoduleDenseNet2D.load_from_checkpoint(ckpt_densenet)
        y_true, y_pred, y_probs = evaluate_model(model_densenet, val_loader)
        results['DenseNet121'] = (y_true, y_pred, y_probs)
        print_metrics('2D DenseNet121', y_true, y_pred, y_probs)
        
        # Plot the individual confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Purples', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix (2D DenseNet121)')
        plt.savefig(os.path.join(out_dir, '2d_densenet_cm.png'), dpi=300)
        plt.close()

    # 4. Plot the combined ROC comparison curve
    if len(results) == 2:
        plt.figure(figsize=(7, 6))
        colors = {'ResNet18': 'darkorange', 'DenseNet121': 'purple'}
        for name, (y_t, y_p, y_prob) in results.items():
            fpr, tpr, _ = roc_curve(y_t, y_prob)
            auc = roc_auc_score(y_t, y_prob)
            plt.plot(fpr, tpr, color=colors[name], lw=2, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('2D Models ROC Comparison (Max Cross-Section)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '2d_combined_roc.png'), dpi=300)
        plt.close()
        print(f"\n✅ Combined 2D ROC comparison curve generated!")