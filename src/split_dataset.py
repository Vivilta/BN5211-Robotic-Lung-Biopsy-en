import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 🛠️ Configuration loading
# ==========================================
def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================
# 🚀 Step 8: Manual stratified split with minimum class representation.
# ==========================================
if __name__ == "__main__":
    cfg = load_config()
    processed_dir = Path(cfg['data']['processed_dir'])
    
    manifest_path = processed_dir / "final_dataset_manifest.csv"
    if not manifest_path.exists():
        print(f"Error: Dataset manifest not found: {manifest_path}")
        exit()
        
    df = pd.read_csv(manifest_path)
    
    print("==================================================")
    print("⚖️ Starting the manual stratified split for the imbalanced dataset...")
    
    # 1. Extract the highest risk label for each patient.
    patient_df = df.groupby('patient_id')['multi_label'].max().reset_index()
    patient_df.rename(columns={'multi_label': 'patient_max_risk'}, inplace=True)
    print(f"-> Total unique patients: {len(patient_df)}")
    
    # 2. Apply the manual stratified assignment algorithm.
    patient_to_split = {}
    np.random.seed(42) # Fix the seed so every run produces the same split.
    
    for risk_label, group in patient_df.groupby('patient_max_risk'):
        patients = group['patient_id'].values.copy()
        np.random.shuffle(patients) # Shuffle patients within the class.
        
        n = len(patients)
        
        # 💡 For lesion-positive classes (risk > 0) with at least 3 patients, ensure each split gets a sample.
        if risk_label > 0 and n >= 3:
            test_count = max(1, int(np.round(n * 0.2)))  # Ensure at least one patient in the test set.
            val_count = max(1, int(np.round(n * 0.1)))   # Ensure at least one patient in the validation set.
        else:
            # Split the large healthy-control group using the standard proportions.
            test_count = int(np.round(n * 0.2))
            val_count = int(np.round(n * 0.1))
            
        test_pts = patients[:test_count]
        val_pts = patients[test_count : test_count + val_count]
        train_pts = patients[test_count + val_count :]
        
        for p in test_pts: patient_to_split[p] = 'test'
        for p in val_pts:  patient_to_split[p] = 'val'
        for p in train_pts: patient_to_split[p] = 'train'
    
    # 3. Map split assignments back to the source manifest and save.
    df['split'] = df['patient_id'].map(patient_to_split)
    final_split_path = processed_dir / "final_split_train_manifest.csv"
    df.to_csv(final_split_path, index=False, encoding="utf-8-sig")
    
    print("\n==================================================")
    print("📊 Dataset split summary (7:1:2) with minimum class representation")
    print("==================================================")
    print("📌 [1] Original sample distribution:")
    print(df['split'].value_counts().to_string())
    
    print("\n☯️ [2] Binary labels (benign/malignant) by split:")
    print(pd.crosstab(df['split'], df['binary_desc']))
    
    print("\n🔮 [3] Multiclass labels (low/intermediate/high risk) by split:")
    print(pd.crosstab(df['split'], df['multi_desc']))
    print("==================================================")
