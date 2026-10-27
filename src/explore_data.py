import yaml
import pydicom
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 🛠️ Configuration loading
# ==========================================
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the global YAML path configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================
# 🧬 Core feature extraction: parse each reader's spatial annotations and scores.
# ==========================================
def parse_xml_all_annotations(xml_path: str, patient_id: str) -> list:
    """
    Parse one XML file to extract reader scores, 3D center coordinates,
    and morphological features such as sphericity and diameter. These
    coordinates support later clustering of annotations for the same nodule.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []

    annotations = []
    for session in root.iter():
        if session.tag.endswith('readingSession'):
            for nodule in session.iter():
                # Keep explicit nodule annotations with a diameter >= 3 mm.
                if nodule.tag.endswith('unblindedReadNodule'):
                    nodule_id = "Unknown"
                    malignancy = np.nan   
                    sphericity = np.nan   
                    z_position = np.nan   
                    x_coords = []         
                    y_coords = []         
                    
                    for child in nodule:
                        if child.tag.endswith('noduleID'):
                            nodule_id = child.text
                            
                    for child in nodule.iter():
                        if child.tag.endswith('malignancy'):
                            malignancy = int(child.text)  
                        elif child.tag.endswith('sphericity'):
                            sphericity = int(child.text)  
                        elif child.tag.endswith('imageZposition'):
                            z_position = float(child.text) 
                        elif child.tag.endswith('xCoord'):
                            x_coords.append(int(child.text)) 
                        elif child.tag.endswith('yCoord'):
                            y_coords.append(int(child.text)) 
                            
                    if x_coords and y_coords and not np.isnan(z_position):
                        x_center = np.mean(x_coords) 
                        y_center = np.mean(y_coords) 
                        
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        # The average LIDC pixel spacing is approximately 0.7 mm.
                        diameter = max(width, height) * 0.7
                        
                        annotations.append({
                            'patient_id': patient_id,
                            'nodule_id': nodule_id,
                            'malignancy': malignancy,
                            'sphericity': sphericity,
                            'x_center': x_center,
                            'y_center': y_center,
                            'z_position': z_position,
                            'diameter': diameter
                        })
    return annotations

# ==========================================
# 🎯 Physical-space clustering: merge annotations by location, not by name.
# ==========================================
def cluster_patient_nodules(annotations: list, distance_threshold: float = 5.0) -> list:
    """
    Calculate the 3D Euclidean distance between nodule centers marked by
    different readers. Centers within 5.0 mm are treated as the same nodule.
    """
    clusters = [] 
    for ann in annotations:
        placed = False
        for cluster in clusters:
            ref = cluster[0] 
            
            dx = (ann['x_center'] - ref['x_center']) * 0.7  
            dy = (ann['y_center'] - ref['y_center']) * 0.7  
            dz = ann['z_position'] - ref['z_position']      
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            if distance <= distance_threshold:
                cluster.append(ann)
                placed = True
                break
        
        if not placed:
            clusters.append([ann])
            
    golden_nodules = []
    for cluster in clusters:
        scored_anns = [a for a in cluster if not np.isnan(a['malignancy'])]
        if not scored_anns: continue
        
        reader_count = len(scored_anns)                         
        mean_malignancy = np.mean([a['malignancy'] for a in scored_anns]) 
        mean_diameter = np.mean([a['diameter'] for a in cluster])       
        
        mean_sphericity = np.nanmean([a['sphericity'] for a in cluster]) 
        mean_x = np.mean([a['x_center'] for a in cluster])              
        mean_y = np.mean([a['y_center'] for a in cluster])              
        mean_z = np.mean([a['z_position'] for a in cluster])            
        
        golden_nodules.append({
            'patient_id': cluster[0]['patient_id'],
            'nodule_id': cluster[0]['nodule_id'], 
            'reader_count': reader_count,
            'mean_malignancy': mean_malignancy,
            'diameter': mean_diameter,
            'nodule_sphericity': mean_sphericity,
            'pos_x': mean_x,
            'pos_y': mean_y,
            'pos_z': mean_z
        })
    return golden_nodules

# ==========================================
# 🚀 Main workflow
# ==========================================
if __name__ == "__main__":
    cfg = load_config()
    lidc_root = Path(cfg['data']['raw_lidc_dir'])
    output_dir = Path(cfg['data']['processed_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    

    print("==================================================")
    
    # 1. Extract patient demographics (age and sex).
    patient_dict = {}
    print("-> Scanning DICOM headers for age and sex...")
    for patient_folder in lidc_root.glob("LIDC-IDRI-*"):
        patient_id = patient_folder.name
        dcm_files = list(patient_folder.rglob("*.dcm"))
        if dcm_files:
            try:
                ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                age = getattr(ds, 'PatientAge', 'Unknown')
                sex = getattr(ds, 'PatientSex', 'Unknown')
                if age != 'Unknown' and isinstance(age, str) and age.endswith('Y'):
                    age = int(age.replace('Y', '').replace('0', '', 1) if age.startswith('0') else age.replace('Y', ''))
                patient_dict[patient_id] = {'age': age, 'sex': sex}
            except Exception:
                patient_dict[patient_id] = {'age': 'Unknown', 'sex': 'Unknown'}

    # 2. Read spatial annotations from all XML files in the dataset.
    all_xml_files = list(lidc_root.rglob("*.xml"))
    all_patient_annotations = {}
    
    print(f"-> Parsing spatial annotations from {len(all_xml_files)} XML files...")
    for xml_path in all_xml_files:
        patient_id = "Unknown"
        for part in xml_path.parts:
            if "LIDC-IDRI-" in part:
                patient_id = part
                break
        anns = parse_xml_all_annotations(str(xml_path), patient_id)
        if patient_id not in all_patient_annotations:
            all_patient_annotations[patient_id] = []
        all_patient_annotations[patient_id].extend(anns)
        
    # 3. Cluster nodule annotations in 3D physical space within each patient.
    print("-> Clustering nodules in 3D physical space (merging annotations of the same lesion)...")
    all_clustered_nodules = []
    for pid, anns in all_patient_annotations.items():
        if anns:
            clustered = cluster_patient_nodules(anns, distance_threshold=5.0)
            all_clustered_nodules.extend(clustered)
            
    df_clustered = pd.DataFrame(all_clustered_nodules)
    
    # 4. Add clinical features and create the master data exploration table.
    print("-> Assembling the complete manifest...")
    df_clustered['patient_age'] = df_clustered['patient_id'].map(lambda x: patient_dict.get(x, {'age': 'Unknown'})['age'])
    df_clustered['patient_sex'] = df_clustered['patient_id'].map(lambda x: patient_dict.get(x, {'sex': 'Unknown'})['sex'])
    
    # Set the manifest column order, retaining nodule_id for ROI extraction.
    master_cols = ['patient_id', 'nodule_id', 'patient_age', 'patient_sex', 'diameter', 'nodule_sphericity', 'pos_x', 'pos_y', 'pos_z', 'reader_count', 'mean_malignancy']
    df_master = df_clustered[master_cols].copy()
    
    # 💡 Data dictionary (exported column names are already in English).
    # 'patient_id': patient identifier
    # 'patient_age': patient age
    # 'patient_sex': patient sex
    # 'diameter': physical diameter (mm)
    # 'nodule_sphericity': mean sphericity score (1-5)
    # 'pos_x': spatial X coordinate
    # 'pos_y': spatial Y coordinate
    # 'pos_z': absolute Z coordinate
    # 'reader_count': number of agreeing readers
    # 'mean_malignancy': mean malignancy score (1-5)
    # 'label': final reference label (0 benign / 1 malignant)
    
    master_path = output_dir / "master_data_exploration.csv"
    df_master.to_csv(master_path, index=False, encoding="utf-8-sig")
    print(f"✅ Results saved to: {master_path}")
    
    # 5. Apply the reader-count and diameter filters to create final ROI labels.
    print("\n-> Applying filters: reader count >= 3 and diameter >= 3.0 mm...")
    df_filtered = df_master[(df_master['reader_count'] >= 3) & (df_master['diameter'] >= 3.0)].copy()
    
    def assign_binary_label(score):
        if score > 3.0: return 1   # Malignant
        elif score < 3.0: return 0 # Benign
        else: return -1            # A score of exactly 3 is indeterminate and is excluded.
        
    df_filtered['label'] = df_filtered['mean_malignancy'].apply(assign_binary_label)
    df_final = df_filtered[df_filtered['label'] != -1].copy()
    
    final_label_path = output_dir / "cleaned_nodules_labels.csv"
    df_final.to_csv(final_label_path, index=False, encoding="utf-8-sig")
    

    print("==================================================")
    print(f"-> High-quality nodules meeting all criteria: {len(df_final)}")
    print(f"   - Reference benign nodules (Label 0): {(df_final['label'] == 0).sum()}")
    print(f"   - Reference malignant nodules (Label 1): {(df_final['label'] == 1).sum()}")
    print(f"✅ Results saved to: {final_label_path}")
    print("==================================================")
