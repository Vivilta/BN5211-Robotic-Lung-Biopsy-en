import os
import glob
import torch
import numpy as np

# Import the core network model.
from train_3D_attention import LitAttentionModel

class LungNodulePredictor:
    """
    Inference API for the 3D pulmonary nodule deep learning model.
    Supports single predictions (predict_single) and batch processing (predict_batch).
    """
    def __init__(self, checkpoint_dir="models/checkpoints_attention/", device=None):
        """
        Initialize the inference engine, load the latest checkpoint, and select a device.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load the latest checkpoint automatically.
        ckpts = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}.")
        ckpts.sort(key=os.path.getmtime)
        best_ckpt = ckpts[-1]

        print(f"⚙️ [API Init] Loading model checkpoint: {os.path.basename(best_ckpt)}")
        print(f"⚙️ [API Init] Compute device: {self.device}")

        self.model = LitAttentionModel.load_from_checkpoint(best_ckpt)
        self.model.to(self.device)
        self.model.eval() # Switch to inference mode.

        self.classes = {0: "Benign", 1: "Malignant"}

    def preprocess(self, npy_path):
        """
        Load a NumPy array and convert it to a tensor with shape (1, 1, 32, 32, 32).
        """
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"File not found: {npy_path}")

        cube = np.load(npy_path)
        if cube.shape != (32, 32, 32):
            raise ValueError(f"Invalid input array shape: expected (32, 32, 32), got {cube.shape}")

        # (D, H, W) -> (B, C, D, H, W) = (1, 1, 32, 32, 32)
        tensor = torch.tensor(cube, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_single(self, npy_path):
        """
        Run inference on a single sample.
        :return: A dictionary containing the predicted class and malignancy probability.
        """
        input_tensor = self.preprocess(npy_path)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class_idx = np.argmax(probs)
        malignancy_prob = float(probs[1])

        return {
            "file": os.path.basename(npy_path),
            "prediction_label": self.classes[pred_class_idx],
            "malignancy_probability": round(malignancy_prob, 4),
            "is_high_risk": malignancy_prob > 0.5
        }

    @torch.no_grad()
    def predict_batch(self, npy_path_list):
        """
        Run inference on a batch of samples.
        :return: A list of inference result dictionaries.
        """
        results = []
        for path in npy_path_list:
            res = self.predict_single(path)
            results.append(res)
        return results

# ==========================================
# 🚀 API usage example.
# ==========================================
if __name__ == "__main__":
    print("=== 🚀 [Task 8] Model Inference API Test ===")
    try:
        # Initialize the inference engine.
        predictor = LungNodulePredictor()

        # Find a few test samples.
        test_dir = "data/processed/all_roi_cubes/"
        sample_files = glob.glob(os.path.join(test_dir, "*.npy"))[:3]

        if len(sample_files) > 0:
            print("\n🔹 Test 1: Single Inference (predict_single)")
            single_res = predictor.predict_single(sample_files[0])
            print(single_res)

            print("\n🔹 Test 2: Batch Inference (predict_batch)")
            batch_res = predictor.predict_batch(sample_files)
            for r in batch_res:
                print(f"  -> File: {r['file']}, Prediction: {r['prediction_label']}, Malignancy probability: {r['malignancy_probability']*100:.2f}%")
        else:
            print("No .npy test samples found. Check the dataset path.")

    except Exception as e:
        print(f"API test failed: {e}")
