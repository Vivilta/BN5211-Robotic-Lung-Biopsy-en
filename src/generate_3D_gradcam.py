import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataset_3D import LungNodule3DDataset
from train_3D_attention import LitAttentionModel

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, target_class]
        score.backward() 
        weights = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

def process_cam_slice(img_slice, cam_slice):
    img_uint8 = np.uint8(255 * img_slice)
    img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    heatmap = np.uint8(255 * cam_slice)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(heatmap, 0.4, img_color, 0.6, 0)
    return img_color, heatmap, overlay

if __name__ == "__main__":
    out_dir = "models/evaluation_results/grad_cam/"
    os.makedirs(out_dir, exist_ok=True)
    
    # Load the model.
    ckpt_dir = "models/checkpoints_attention/"
    ckpts = glob.glob(os.path.join(ckpt_dir, "*cbam*.ckpt"))
    ckpts.sort(key=os.path.getmtime)
    lit_model = LitAttentionModel.load_from_checkpoint(ckpts[-1]).eval()
    grad_cam = GradCAM3D(lit_model.model, lit_model.model.layer2[-1])
    
    dataset = LungNodule3DDataset("data/processed/final_dataset_manifest.csv", "data/processed/all_roi_cubes/", transform=None)
    
    # Collect samples for the comparison gallery.
    gallery_data = [] # Stores (Original, Heatmap, Overlay, Label).
    for idx in range(len(dataset)):
        img_tensor, label = dataset[idx]
        is_malignant = (label.item() == 1)
        
        # Select the first two benign and two malignant samples.
        if (is_malignant and len([x for x in gallery_data if x[3]==1]) < 2) or \
           (not is_malignant and len([x for x in gallery_data if x[3]==0]) < 2):
            cam_3d = grad_cam.generate(img_tensor.unsqueeze(0), target_class=label.item())
            center_z = img_tensor.shape[1] // 2
            img_slice = img_tensor[0, center_z, :, :].numpy()
            cam_slice = cam_3d[center_z, :, :]
            gallery_data.append((*process_cam_slice(img_slice, cam_slice), label.item()))
            
        if len(gallery_data) >= 4: break

    # Plot the complete comparison gallery.
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    titles = ["Original CT", "Grad-CAM", "Overlay"]
    
    for row, (orig, heat, over, label) in enumerate(gallery_data):
        label_text = "Malignant" if label == 1 else "Benign"
        axes[row, 0].imshow(orig)
        axes[row, 1].imshow(heat)
        axes[row, 2].imshow(over)
        axes[row, 0].set_ylabel(f"{label_text} Sample {row+1}", fontsize=12, fontweight='bold')
        if row == 0:
            for col, title in enumerate(titles): axes[0, col].set_title(title, fontsize=14)
        for col in range(3): axes[row, col].axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "GradCAM_Comparison_Gallery.png"), dpi=300)
    print(f"✅ Complete comparison gallery generated -> {out_dir}GradCAM_Comparison_Gallery.png")
