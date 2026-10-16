import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import os

# ==========================================
# 🛠️ 基础配置加载模块
# ==========================================
def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载全局路径配置文件 yaml"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ==========================================
# 🔍 核心工具：直接从 LIDC 的深层目录中解出 DICOM 三维图像
# ==========================================
def load_lidc_dicom_series(patient_dir: str):
    """
    暴力递归扫描：不管文件夹嵌套多深，
    只要找到包含 .dcm 的目录，直接读入为 3D 图像矩阵
    """
    for root, dirs, files in os.walk(patient_dir):
        if any(f.endswith('.dcm') for f in files):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(root)
            if dicom_names:
                reader.SetFileNames(dicom_names)
                try:
                    return reader.Execute()
                except:
                    continue
    return None

# ==========================================
# ✂️ 核心工具：裁剪 3D 局部图像块 (ROI)
# ==========================================
def extract_3d_cube(itk_image, pos_x, pos_y, pos_z, cube_size=32):
    """
    根据融合后的三维物理空间坐标，在整体体数据中裁剪出 32x32x32 的局部结节立方体，
    并原地完成医疗标准的 HU 窗口值归一化。
    """
    if cube_size <= 0 or cube_size % 2 != 0:
        raise ValueError("cube_size must be a positive even number")

    img_array = sitk.GetArrayFromImage(itk_image)  # Array order: (Z, Y, X)

    # SimpleITK is authoritative here: it applies origin, spacing, and direction.
    # Treating physical X/Y values as voxel indices silently extracts the wrong lesion
    # whenever the scan origin or spacing is not zero/one.
    physical_point = (float(pos_x), float(pos_y), float(pos_z))
    x_index, y_index, z_index = itk_image.TransformPhysicalPointToIndex(physical_point)

    half_size = cube_size // 2
    z_start, z_end = z_index - half_size, z_index + half_size
    y_start, y_end = y_index - half_size, y_index + half_size
    x_start, x_end = x_index - half_size, x_index + half_size

    max_z, max_y, max_x = img_array.shape
    if z_start < 0 or y_start < 0 or x_start < 0 or z_end > max_z or y_end > max_y or x_end > max_x:
        return None

    cube = img_array[z_start:z_end, y_start:y_end, x_start:x_end]

    # Lung-window normalization used by the supplied 32^3 checkpoints.
    cube = np.clip(cube, -1000.0, 400.0)
    cube = (cube + 1000.0) / 1400.0
    return cube.astype(np.float32)

# ==========================================
# 🚀 多维标签并行挂载与 3D ROI 物理裁切
# ==========================================
if __name__ == "__main__":
    cfg = load_config()
    lidc_root = Path(cfg['data']['raw_lidc_dir'])
    processed_dir = Path(cfg['data']['processed_dir'])
    
    # 加载上游洗好的黄金结节清单
    labels_path = processed_dir / "cleaned_nodules_labels.csv"
    if not labels_path.exists():
        print(f"【错误】找不到基础清洗标签表，请先确保重新运行了 explore_data.py！")
        exit()
    df_labels = pd.read_csv(labels_path)
    
    # 建立 3D 局部矩阵（.npy 文件）存放仓库
    cube_store_dir = processed_dir / "all_roi_cubes"
    cube_store_dir.mkdir(parents=True, exist_ok=True)
    
    # 扫描全量原始病例文件夹
    all_lidc_patients = [f for f in lidc_root.glob("LIDC-IDRI-*") if f.is_dir()]
    
    print("==================================================")
    print("🔪 正在执行 3D 局部切图与双标签并行挂载...")
    print(f"-> 共检测到 LIDC 原始患者数据: {len(all_lidc_patients)} 例")
    print("==================================================")
    
    final_dataset_records = []
    roi_success_count = 0
    
    for patient_folder in all_lidc_patients:
        patient_id = patient_folder.name
        
        # 检索该病人在黄金名单里有没有合格的物理结节
        associated_nodules = df_labels[df_labels['patient_id'] == patient_id]
        
        # 情况 A：如果有明确的高质量病灶，启动加载与切割
        if not associated_nodules.empty:
            itk_image = load_lidc_dicom_series(str(patient_folder))
            
            for _, n_row in associated_nodules.iterrows():
                nodule_id = n_row['nodule_id']
                mean_score = n_row['mean_malignancy']
                
                # 抓取聚类好的三维物理位置
                pos_x = n_row['pos_x']
                pos_y = n_row['pos_y']
                pos_z = n_row['pos_z']
                
                # 挂载二分类标签
                if mean_score > 3.0: binary_label, binary_desc = 1, "恶性病灶"
                elif mean_score < 3.0: binary_label, binary_desc = 0, "良性病灶"
                else: binary_label, binary_desc = -1, "模糊争议"
                
                # 挂载多分类风险标签
                if mean_score <= 2.5: multi_label, multi_desc = 0, "低风险"
                elif mean_score <= 3.5: multi_label, multi_desc = 1, "中风险"
                else: multi_label, multi_desc = 2, "高风险"
                
                cube_path_str = "未提取或提取失败"
                if itk_image is not None:
                    roi_cube = extract_3d_cube(itk_image, pos_x, pos_y, pos_z)
                    if roi_cube is not None:
                        cube_filename = f"{patient_id}_{str(nodule_id)[:8]}.npy"
                        save_path = cube_store_dir / cube_filename
                        np.save(save_path, roi_cube)
                        cube_path_str = str(save_path)
                        roi_success_count += 1
                        print(f"✅ 成功截获 3D ROI: {patient_id} - 结节 {nodule_id}")
                        
                final_dataset_records.append({
                    'patient_id': patient_id,
                    'data_type': '含高质量结节',
                    'nodule_id': str(nodule_id),
                    'cube_file_path': cube_path_str,
                    'mean_score': mean_score,
                    'binary_label': binary_label,
                    'binary_desc': binary_desc,
                    'multi_label': multi_label,
                    'multi_desc': multi_desc
                })
                
        # 情况 B：完全健康的阴性对照组
        else:
            final_dataset_records.append({
                'patient_id': patient_id,
                'data_type': '阴性健康对照',
                'nodule_id': '全肺健康(无结节)',
                'cube_file_path': '无需切割3D矩阵',
                'mean_score': 0.0,
                'binary_label': 0, 
                'binary_desc': '全阴性健康对照(良性)',
                'multi_label': 0,   
                'multi_desc': '无病灶极低风险'
            })
            
    # 固化标准英文大账本给第八步读取
    df_final_manifest = pd.DataFrame(final_dataset_records)
    manifest_csv_path = processed_dir / "final_dataset_manifest.csv"
    df_final_manifest.to_csv(manifest_csv_path, index=False, encoding="utf-8-sig")
    

    print("==================================================")
    print(f"-> 大账本已成功收录总行数: {len(df_final_manifest)} 行记录")
    print(f"-> 成功在 3D 空间切下并保存的黄金结节立方体: {roi_success_count} 个")
    print(f"✅ 结果已保存至: {manifest_csv_path}")
    print("==================================================")
