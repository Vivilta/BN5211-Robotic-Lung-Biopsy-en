# BN5211 肺结节影像分析与机器人穿刺仿真

本项目覆盖肺结节 CT 数据处理、恶性风险二分类模型、模型评估与可视化、交互式推理，以及后续的 UR5 穿刺路径规划和 CoppeliaSim 仿真。整体链路如下：

```text
LIDC-IDRI / LUNA16 数据
  -> XML/CSV 标注整理
  -> CT 预处理与结节 ROI 构建
  -> 患者数据集清单与训练/验证划分
  -> 2D/3D 模型训练和比较
  -> 独立数据评估、亚组/集成/阈值分析、Grad-CAM
  -> Streamlit ROI 推理演示
  -> [机器人扩展] CT 几何 -> 避障规划 -> UR5 IK -> CoppeliaSim 仿真
```

模型输出和机器人流程均为课程科研原型，不是临床诊断或治疗系统，不可用于真实患者诊疗或控制真实手术机器人。Streamlit 页面目前包含固定的风险区间和建议文案；这些内容不是临床验证结果，也不是治疗建议。

## 1. 项目模块

### 1.1 数据与模型主链

- **LIDC-IDRI**：DICOM CT、XML 放射科医生标注；整理医生评分并提取局部 3D 结节 ROI。
- **LUNA16**：MHD CT 和 `annotations.csv`；用于全肺预处理、ROI 提取及跨数据集评估。
- **模型**：2D ResNet/DenseNet/CBAM，3D ResNet/CBAM/VNet，以及无增强消融实验。
- **分析**：分类指标、LUNA16 评估、临床亚组分析、模型集成、ROC/Youden 工作点和 Grad-CAM 可视化。
- **交互演示**：Streamlit 页面接受预先标准化的 `.npy` ROI，显示概率和风险分层。

### 1.2 机器人扩展

机器人链路复用训练好的 3D CBAM checkpoint：由已登记病例的 ROI、CT 和 target 生成局部骨候选 AABB，规划直线路径，计算 UR5 分析 IK，并由 CoppeliaSim UR5、simIK 和 ZeroMQ Remote API 执行运动学仿真。它是主项目后续的机器人演示扩展，不替代前面的数据和模型链路。当前病例配置启用了 `force_robot_demo`，因此低于恶性概率阈值时也会继续做软件集成规划；模型分数不会自动选择结节，也不是机器人规划的安全门控。

## 2. 环境和本地数据

推荐 Python 3.9。安装依赖：

```bash
conda create -n bn5211 python=3.9 -y
conda activate bn5211
python -m pip install -r requirements.txt
```

配置文件 [config/config.yaml](config/config.yaml) 中的数据根目录默认为：

```yaml
data:
  raw_lidc_dir: "./data/LIDC-IDRI"
  raw_luna_dir: "./data/LUNA16"
  processed_dir: "./data/processed"
```

将路径改成你本机的数据位置，或将数据放到上述目录结构。原始医学影像、训练 checkpoint、处理后的完整数据通常不纳入 Git；每位运行者都需要准备相应本地资产。

基础依赖检查：

```bash
python -c "import torch, SimpleITK, pydicom, numpy, pandas; print(torch.__version__)"
```

## 3. 目录和主要产物

```text
config/config.yaml                 # 原始数据与处理数据根目录
configs/cases.json                 # 通用病例登记表（当前含示例病例）
src/preprocess.py                  # LUNA16 MHD 全肺重采样/掩膜/归一化
src/explore_data.py                # LIDC XML 标注整理与标签清单
src/extract_roi.py                 # LIDC 3D ROI 提取
src/extract_luna16.py              # LUNA16 结节 ROI 与 manifest
src/split_dataset.py               # 患者级 split manifest 生成
src/train_*.py                     # 2D/3D 模型训练
src/evaluate_*.py                  # 评估、解释性及应用分析
src/run_case_pipeline.py           # 单病例机器人流程编排
data/processed/                    # 本地生成的 ROI、CSV manifest 和预处理影像
models/                             # 本地 checkpoint、日志及分析图
data/demo_nodules/                  # 机器人流程 JSON、预览图和运行记录
scenes/                             # CoppeliaSim 场景文件
```

原始影像、预处理矩阵、checkpoint 和 `data/` 下的运行结果不随仓库分发。仓库包含基础/历史 CoppeliaSim 示例场景；每个新病例的场景和运行结果仍需在本机生成。`configs/cases.json` 当前含本机路径，换电脑运行前需按后文更新。

## 4. 全链路运行指南（数据到模型）

以下命令都从仓库根目录运行。先确认 `config/config.yaml` 的路径正确，并准备 LIDC-IDRI、LUNA16 对应输入资产。

### 从原始数据开始：推荐执行顺序

如果要从头准备本项目的 LIDC-IDRI 和 LUNA16 数据产物，按以下顺序运行。该顺序把两套数据的准备步骤放在同一条主线中：

```bash
python src/preprocess.py
python src/explore_data.py
python src/extract_roi.py
python src/extract_luna16.py
python src/split_dataset.py
```

运行前需要准备：

- LIDC-IDRI 原始 DICOM 和 XML，目录位置与 `config/config.yaml` 的 `raw_lidc_dir` 一致。
- LUNA16 原始 `.mhd`/配套影像文件，以及 `annotations.csv`。目前 `preprocess.py` 会读取配置中的 `raw_luna_dir` 和 `processed_dir`；但 `extract_luna16.py` 将输入/输出路径写为相对路径 `data/LUNA16/` 和 `data/processed/`，因此应从仓库根目录运行，并将 LUNA16 数据放在对应目录，或先按本机目录修改该脚本路径。
- 足够的磁盘空间。`preprocess.py` 会为每个 LUNA16 全肺体数据写出预处理 `.npy`，并非只生成结节小 ROI。

这五步的输入、输出与数据集注意事项见下方各阶段说明。若只处理单个数据集，可以跳过另一数据集专属步骤；但 `split_dataset.py` 依赖 LIDC 的 `final_dataset_manifest.csv`，因此仅准备 LUNA16 时不能直接执行它来生成本项目的训练划分。

### 步骤 1：LUNA16 全肺预处理

确认 `config/config.yaml` 中的 `raw_luna_dir` 指向含 `.mhd` 和配套影像文件的 LUNA16 目录，然后运行：

```bash
python src/preprocess.py
```

脚本对每个全肺体数据进行 1 mm 各向同性重采样、阈值/形态学掩膜处理和 HU 归一化，输出：

```text
data/processed/preprocessed_cubes/*_preprocessed.npy
```

处理全量数据需要较多时间和磁盘空间；脚本支持通过已存在的输出文件跳过病例。

### 步骤 2：LIDC 标注整理

```bash
python src/explore_data.py
```

预期生成：

```text
data/processed/master_data_exploration.csv
data/processed/cleaned_nodules_labels.csv
```

该脚本读取 LIDC XML，汇总结节空间坐标和评分，再按医生数、直径和平均恶性评分筛选标签。它是项目的数据整理脚本，不应把输出称为临床金标准。代码当前使用简化坐标聚类/直径近似，使用其结果前需抽样核对。

### 步骤 3：生成 LIDC 结节 ROI 和数据清单

```bash
python src/extract_roi.py
```

预期生成：

```text
data/processed/all_roi_cubes/*.npy
data/processed/final_dataset_manifest.csv
```

每个 `.npy` 是单通道输入的 `32 x 32 x 32` CT patch，按脚本中的窗口裁剪和归一化约定生成。运行前后都要检查 ROI/标签/原始 DICOM 的对应关系。**当前 LIDC ROI 脚本会在患者目录下遍历并使用找到的第一套可读 DICOM 序列**；一个患者有多个序列时不能默认认为它就是 XML 标注所对应的序列。对需要严格病例追踪的数据，应先确认序列与 ROI 关联。

### 步骤 4：生成 LUNA16 结节 ROI

确认 `data/LUNA16` 下存在原始 `.mhd` 数据、配套影像文件及官方 `annotations.csv`，并已完成步骤 1，然后运行：

```bash
python src/extract_luna16.py
```

脚本生成：

```text
data/processed/luna16_roi_cubes/*.npy
data/processed/luna16_dataset_manifest.csv
```

提取脚本根据 `annotations.csv` 里的物理坐标裁切 ROI。LUNA16 的标注/标签口径与 LIDC 不同，不能不加说明地将两个数据集的标签解释为同一临床结局。

### 步骤 5：生成数据划分清单

```bash
python src/split_dataset.py
```

生成 `data/processed/final_split_train_manifest.csv`。该脚本按患者和风险分层生成 split 清单。不过当前多数 `train_*.py` 和 `evaluate_*.py` 脚本自行用 `random_split` 划分数据，并不读取这个 split manifest。因此，**目前不能仅凭运行 `split_dataset.py` 就声称所有训练/评估都采用了患者级独立划分**；正式报告模型性能前应核实并统一训练、验证和测试索引，避免患者/结节泄漏。

### 阶段五：训练模型（可选）

如果只需复用已提供的 CBAM 权重进行推理或机器人演示，可跳过本阶段。若重新训练，先完成前述 ROI 和 manifest 生成，再按实验目标执行模型脚本。

```bash
# 2D 基线与 DenseNet
python src/train_2D.py
python src/train_2D_densenet.py
python src/train_2D_attention.py

# 3D 基线、CBAM、VNet 与消融
python src/train_3D.py
python src/train_3D_attention.py
python src/train_3D_vnet.py
python src/train_3D_ablation.py
```

训练脚本使用 PyTorch Lightning，按 `val_acc` 保存最佳 checkpoint 到各自的 `models/checkpoints*` 子目录，并写 TensorBoard 日志到 `models/logs/`。脚本超参数、训练集划分和保存目录以代码为准；运行前建议核对数据切分和随机种子设置。机器人当前使用的模型架构是 `src/train_3D_attention.py` 中的 3D CBAM，不能拿普通 3D ResNet 或 2D checkpoint 替代。

### 阶段六：模型评估和分析（可选）

```bash
# 2D / 3D 模型对比
python src/evaluate_2D_models.py
python src/evaluate_3D_models.py

# LUNA16 跨数据集评估（要求已准备 LUNA16 manifest、ROI 和相应 checkpoint）
python src/evaluate_luna16_generalization.py

# 亚组、集成和阈值分析
python src/evaluate_clinical_subgroups.py
python src/evaluate_ensemble.py
python src/evaluate_clinical_value.py

# 3D CBAM Grad-CAM
python src/generate_3D_gradcam.py
```

评估图表主要写入 `models/evaluation_results/`。这些脚本是课程项目中的实验分析工具；其输出受数据划分、标签定义、样本数量和实现方式影响，不代表外部临床验证或临床阈值。

### 阶段七：Streamlit 推理演示（可选）

`src/app.py` 接收已经按项目预处理约定生成的 `.npy` ROI，而不是原始 DICOM。页面默认从 `models/checkpoints_attention/` 加载 3D CBAM 权重。安装 `requirements.txt` 中的依赖并将兼容架构的 checkpoint 放入该目录后运行：

```bash
streamlit run src/app.py
```

上传 ROI 后可查看模型概率和页面中定义的演示风险分层。当前风险边界及处置建议文案是固定在页面代码中的展示逻辑，未经临床验证；它不负责从 CT 自动检测、定位或裁切结节，不应据此作任何诊疗决定。

## 5. 机器人穿刺仿真扩展

机器人部分接收一个已经确认的 ROI/target/CT 组合。LIDC XML 中有多个阅片者和多个结节，脚本不会只凭患者 ID 猜测仿真目标；需要先选定一个 XML 标注及其阅片者/结节 ID。

### 5.1 病例输入和登记

每个病例登记在通用文件 `configs/cases.json` 的 `cases` 数组中。文件名表示病例登记表，不限定其中病例数量或编号：

```json
{
  "case_id": "LIDC-IDRI-00XX",
  "roi_path": "/absolute/path/to/case_roi.npy",
  "ct_series_path": "/absolute/path/to/one/exact/DICOM/series",
  "series_instance_uid": "UID_READ_FROM_THAT_SERIES",
  "target_voxel": [X, Y, Z]
}
```

字段来源和当前可自动化程度：

| 字段 | 如何取得 | 当前能否可靠自动生成 |
| --- | --- | --- |
| `case_id` | LIDC 病例目录名，例如 `LIDC-IDRI-0035` | 可以；从输入数据目录名取得 |
| `roi_path` | 从 `data/processed/final_dataset_manifest.csv` 的 `cube_file_path` 找到对应 `.npy`，或填写已准备好的 checkpoint 输入 ROI | 可以列出候选 ROI；需核验它确实是要仿真的结节 |
| `ct_series_path` | 指向该病例中唯一、已确认的 DICOM Series 文件夹 | 可离线扫描并列出候选序列；不能仅凭患者 ID 自动选对 |
| `series_instance_uid` | 读取所选序列 DICOM header 的 `SeriesInstanceUID` | 选定序列后可自动读取，并由 pipeline 校验 |
| `target_voxel` | 对应 ROI 中心在所选 CT 序列的 `[x, y, z]` 体素索引；数组内部顺序为 `[z, y, x]` | 当前不能从已有 manifest 可靠恢复，必须核对 ROI 与 CT 的对应关系 |

可以在离线环境中用新增工具从选定的 LIDC XML 标注生成 32³ ROI 和登记元数据。先列出 XML 中的结节、阅片者和评分：

```bash
python src/register_lidc_case.py \
  --annotation-xml /path/to/one/LIDC-annotation.xml \
  --list-annotations
```

然后明确指定同一病例的精确 DICOM series、`noduleID`，以及 XML 中该 ID 若重复时对应的 `servicingRadiologistID`：

```bash
python src/register_lidc_case.py \
  --case-id LIDC-IDRI-00XX \
  --ct-series /path/to/exact/DICOM-series \
  --annotation-xml /path/to/one/LIDC-annotation.xml \
  --nodule-id "Nodule 001" \
  --reader-id "READER_ID" \
  --output-dir data/registered_rois \
  --cases-config configs/cases.json
```

脚本通过 XML 的 `SeriesInstanceUid`、每张轮廓的 `imageSOP_UID` 和 DICOM header 交叉校验序列/切片，再按 DICOM 方向、像素间距和层位置把轮廓中心映射到 `[x,y,z]` 体素。输出包括 checkpoint 使用的 `32 x 32 x 32` HU 窗口归一化 ROI、包含来源和坐标的 registration JSON、三视图 QC PNG，以及可选追加到 `cases.json` 的病例条目。已有输出不会静默覆盖；确需覆盖 ROI 文件时使用 `--overwrite`，替换配置内已存在病例时使用 `--replace-case`。

登记脚本自动化的是“一个明确选中的阅片者标注 → 对应 DICOM → ROI/target/config 记录”，不是从患者号自动选出临床共识结节。LIDC 的不同阅片者结节 ID 不能仅凭相同编号视作同一结节；生成后仍须查看 QC 图确认 ROI 落在预期病灶。原始 DICOM/XML 及 checkpoint 需已在本机下载，脚本不请求在线服务。旧 `extract_roi.py` 仍用于课程训练数据集切片，其“患者目录下第一套可读序列”逻辑不应被用于严格的病例登记。

`patient_to_robot_base` 是 CT 患者坐标到 UR5 base 的 4x4 变换。`configs/cases.json` 中的矩阵是仿真摆位参数，不是实测标定，不能从 CT/XML 推导。真实机器人应用需通过外部标定流程取得矩阵并登记为 `patient_to_robot_transform_path`；本项目仅用于明确标记的教学坐标系演示。checkpoint、输出目录和规划参数在共享 `defaults` 中；病例专属路径、UID、目标体素和演示变换在对应 case 条目中。配置中的绝对路径需按运行电脑修改，相对 ROI 路径按仓库根目录解析。

`src/run_case_pipeline.py` 会检查 ROI 形状/数值、DICOM Series UID 和目标体素到物理坐标的转换，但这些一致性检查不能证明操作者选对了结节或 ROI。新病例应新增一条已核实的 case 记录；障碍物、计划和仿真结果会按病例 ID 单独输出，不要复制其他病例的 target 或结果。

### 5.2 离线生成障碍物和轨迹

```bash
python src/run_case_pipeline.py \
  --config configs/cases.json \
  --case-id LIDC-IDRI-0031 \
  --output-dir data/case_runs/LIDC-IDRI-0031
```

这一步读取 CT、提取局部高密度骨候选 AABB、生成 QC 预览、加载 CBAM checkpoint 推理、搜索直线路径、计算 UR5 分析 IK，并生成：

```text
data/case_runs/LIDC-IDRI-0031/rib_obstacles_0031_segmented.json
data/case_runs/LIDC-IDRI-0031/rib_obstacles_0031_segmented_preview.png
data/case_runs/LIDC-IDRI-0031/planned_trajectory_0031.json
data/case_runs/LIDC-IDRI-0031/pipeline_summary_0031.json
```

0031 是仓库配置中已有完整运动学仿真通过记录的示例病例。0036 在当前配置的 30° 入路范围内，经完整针体扫掠检查后没有安全直线路径；0039 曾有离线规划结果，但不能据此认定当前版本仿真通过。当前 patient obstacles 是局部 HU 阈值生成的分段 AABB 近似，不是完整胸廓/肋骨分割，也未包含真实皮肤表面、血管、软组织和呼吸运动。

### 5.3 生成场景并执行 CoppeliaSim 仿真

启动 CoppeliaSim ZeroMQ Remote API 服务（端口 `23000`）。macOS 示例：

```bash
export COPPELIASIM_ROOT="/Applications/coppeliaSim.app"
"$COPPELIASIM_ROOT/Contents/MacOS/coppeliaSim" \
  -GzmqRemoteApi.rpcPort=23000
```

保持 CoppeliaSim 运行，在仓库另一个终端运行：

```bash
PYTHONPATH="$COPPELIASIM_ROOT/Contents/Resources/programming/zmqRemoteApi/clients/python/src:$PWD/src" \
python src/run_case_pipeline.py \
  --config configs/cases.json \
  --case-id LIDC-IDRI-0031 \
  --run-simulation \
  --frame-delay 0.08 \
  --output-dir data/case_runs/LIDC-IDRI-0031 \
  --scene-output scenes/robotic_lung_biopsy_0031_latest.ttt
```

默认教学场景使用 `220 mm` 针具并将 overhead mount 抬高 `200 mm`，同时补偿 demo 坐标变换，使 CT 患者几何在世界坐标中不移动；它会改变机械臂伸展姿态，不改变针路相对患者的位置。局部障碍提取默认使用 `segment_slices=5` 和 `min_volume_voxels=40`。规划器按 4 mm 针半径加 2 mm 额外余量扩张骨候选 AABB，并检查从预接近到靶点的针体扫掠。场景生成阶段还会对 `approach -> entry -> target` 的整段笛卡尔轨迹逐帧执行 simIK，并检查 UR5/针具与患者障碍、针具与机械臂自碰撞；只有整段预检通过才保存场景。上述数值是仿真假设，不是临床器械规格。ZeroMQ 客户端路径需按本机 CoppeliaSim 安装位置调整。

针长、顶部安装高度和障碍分段参数可以针对一次验证运行覆盖，不会修改配置文件。例如：

```bash
python src/run_case_pipeline.py \
  --config configs/cases.json \
  --case-id LIDC-IDRI-0031 \
  --needle-length-mm 220 \
  --mount-height-mm 200 \
  --segment-slices 5 \
  --min-volume-voxels 40
```

`segment_slices` 表示沿 CT 的 Z 轴（轴位切片方向）把一个连通的骨候选分成多少张切片一组，再为每一组生成一个 AABB 障碍盒。它的单位是 CT 切片数，不是毫米。例如，一个候选跨越约 30 张切片时，`segment_slices=5` 通常会产生约 6 个纵向障碍盒。数值越小，障碍盒越细、数量越多，对局部形状保留更多，但也更容易保留分割噪声；数值越大，障碍盒越少，但包围盒会更粗，可能扩大到肋骨以外的空间。它只影响障碍物近似模型，不改变 ROI、目标点或针的实际插入深度。

当前默认障碍物并不是由训练好的分割 checkpoint 生成的。`src/environment_modeling.py` 直接读取与病例匹配的 DICOM CT，按 HU 阈值（默认 500）进行局部高密度骨候选提取，使用 3D 连通域、体素数过滤和按切片分段，再把每个候选转换成轴对齐包围盒 AABB。`3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt` 只用于 ROI 恶性概率分类，不用于肋骨分割、目标定位或障碍物生成。当前障碍物是仿真用近似，不能称为完整肋骨重建。

本机保存的 0031 和 0040 运行记录各完成了 100/100 个运动学步骤，碰撞检查可用，患者碰撞与针具自碰撞均未报告，最大针尖位置误差约 `0.455 mm`。这是指定软件场景和当前近似障碍模型下的运动学仿真记录，不是动力学验证、真实机器人验证或临床安全证明。0036 当前配置下没有可行路径；0039 没有当前版本的完整通过记录。`data/` 被 Git 忽略，因此上述运行 JSON 不随仓库分发，其他电脑需按命令重新生成并检查结果。

仅当运行记录中碰撞检查可用、`collision_free` 和 `needle_self_collision_free` 均为 true、无碰撞帧、运动阶段完整且针尖误差符合项目容差时，才记录为该软件场景仿真通过。碰撞或 IK 失败就是失败结果，不应通过随意改针长、关闭碰撞或放宽未经论证的约束来凑成功。当前针长参数是几何占位值，未按具体临床器械 IFU 选型。

### 5.4 人工验收清单

每次更换病例、ROI、DICOM 序列、坐标变换或仿真参数后，都应重新验收；不能只依据规划器的“安全轨迹数”或进程退出码判定通过。

1. **输入和坐标**：核对病例 ID、ROI、DICOM `SeriesInstanceUID`、ROI 三视图 QC 图和目标位置；确认 `cases.json` 中的 target voxel 与所选序列一致。病例登记的患者到 UR5 变换目前是教学摆位，不是实测标定。
2. **障碍物预览**：打开 `rib_obstacles_<case>_segmented_preview.png`，确认目标附近的骨候选大致对应 CT 高密度结构，检查明显游离的分割噪声。局部 AABB 是简化模型，预览不能证明肋骨分割完整或临床准确。
3. **轨迹和场景**：检查 `planned_trajectory_<case>.json` 中 approach、entry、target、方向、插入深度与目标一致；打开生成的 `.ttt`，确认顶置 UR5、安装板、针具方向、目标点和障碍物位置合理，没有初始自碰撞或针具穿过机械臂。运行时在 CoppeliaSim 中观察整段 approach、entry、insertion 是否连续且没有非计划的大幅摆动。当前脚本不自动生成 GIF 或视频。
4. **仿真结果 JSON**：检查 `coppeliasim_run_<case>_segmented.json` 的 `executed_steps` 是否等于配置的步数、`collision_check_available` 与 `needle_self_collision_check_available` 是否为 true、`collision_free` 与 `needle_self_collision_free` 是否为 true、`collision_frames` 和 `needle_self_collision_frames` 是否为空，并检查最大/最终针尖误差。
5. **汇总结果**：检查 `pipeline_summary_<case>.json` 指向的输入、障碍、计划、场景和仿真结果均属于同一病例及本次输出目录。任一碰撞、IK 失败、序列不匹配或输入来源不清均判为失败，先查明原因再重跑，不要将失败结果描述为成功穿刺。

人工验收只用于检查软件仿真的一致性和可解释性。当前没有完整患者胸廓、软组织、器官运动、动力学或真实机器人闭环，因此画面中针尖到达规划靶点不等同于完成了临床穿刺活检。

## 6. 当前边界与复现注意事项

- **数据路径**：`config/config.yaml`、机器人配置和若干旧脚本含本地相对/绝对路径；换电脑时需检查并调整。
- **病例到 ROI 的绑定**：`src/register_lidc_case.py` 已将明确选定的 LIDC XML 标注、精确 DICOM series、ROI 和 provenance 封装为可重跑流程；它不会自动替阅片者选择临床共识结节，QC 图和标注选择仍需人工确认。
- **模型验证**：已有训练与评估脚本，但 split manifest 并未被多数训练/评估入口统一使用；指标应谨慎解释，严谨对比前需要统一患者级 split。
- **LUNA16 标签**：LUNA16 标注提供结节中心/直径，不等同于 LIDC 放射科医生恶性评分；当前 LUNA16 提取脚本中的标签构造是项目实现选择，不应当作病理真值。
- **机器人几何**：局部 AABB 只支持软件级碰撞近似；完整肋骨、体表、软组织、运动和真实器械几何尚未建齐。
- **坐标标定**：病例示例的 CT-to-UR5 transform 是教学摆位，真实患者到机械臂的实测标定尚缺。
- **针具参数**：仿真几何长度不能代表临床器械长度；需先选定器械并依据其 IFU 建模。
- **临床范围**：所有模型概率、风险分层和机器人轨迹均为研究演示，不提供临床决策或可执行手术方案。

## 7. 常见问题

**找不到输入文件/manifest**：检查 `config/config.yaml` 和数据目录；按前述阶段先生成所需 CSV、ROI 和预处理文件。

**模型 checkpoint 结构不匹配**：确认 2D、3D ResNet、3D CBAM、VNet checkpoint 与加载类对应；机器人 pipeline 使用 3D CBAM checkpoint。

**机器人病例 ID 未登记**：向机器人 JSON 的 `cases` 数组添加经确认的 ROI、精确 DICOM 序列、Series UID 和 target voxel。

**SeriesInstanceUID 不匹配或扫描到多个序列**：指定唯一准确的 DICOM series 目录并核对配置，不应绕过校验。

**没有无碰撞路径/IK 失败**：检查 target、变换、预览和 UR5 构型；不要为获得成功日志而更改临床器械参数或禁用碰撞检测。

**Remote API 不可用**：确认 CoppeliaSim 已启动且端口为 `23000`，并在 `PYTHONPATH` 中使用该安装自带的 Python ZeroMQ 客户端。

## 8. 验证

运行仓库内的单元测试：

```bash
python -m unittest discover -s tests -v
```

端到端机器人运行还需要本机的 LIDC DICOM、ROI、checkpoint 和已启动的 CoppeliaSim ZeroMQ 服务；单元测试通过不代表病例仿真通过。

## 9. 主要脚本索引

| 功能 | 入口 |
| --- | --- |
| LUNA16 全肺预处理 | `src/preprocess.py` |
| LIDC 标注整理 | `src/explore_data.py` |
| LIDC ROI 和训练 manifest | `src/extract_roi.py` |
| 单病例 XML-DICOM ROI 登记 | `src/register_lidc_case.py` |
| LUNA16 ROI 和 manifest | `src/extract_luna16.py` |
| 患者级 split 清单 | `src/split_dataset.py` |
| 2D/3D 训练 | `src/train_*.py` |
| 模型评估和可解释性 | `src/evaluate_*.py`, `src/generate_3D_gradcam.py` |
| Streamlit 推理 | `src/app.py` |
| 机器人病例编排 | `src/run_case_pipeline.py` |
| CoppeliaSim 场景生成/执行 | `src/create_coppeliasim_scene.py`, `src/coppeliasim_remote.py` |
