# 🫁 肺结节 3D 影像标准化预处理与多维特征构建系统 (LIDC-IDRI / LUNA16)

---

## 1. 项目背景与核心学术贡献 (Project Highlights)

本项目致力于解决肺癌早期筛查中人工阅片耗时长、微小病灶易漏诊的临床痛点。基于 **LIDC-IDRI** 与 **LUNA16** 双数据集，构建了从原始异构医学影像（DICOM/MHD）清洗、特征提取、深度学习建模到临床辅助决策的端到端全栈流水线。

- **多流派网络架构矩阵**：完整复现并横向对比了 2D (ResNet, DenseNet) 与 3D (ResNet, VNet) 医学图像分类基线，深度论证了空间立体特征在结节鉴别中的不可替代性。
- **混合注意力机制与消融验证**：自主研发并注入 2D/3D Attention 模块（如 3D CBAM），并通过严谨的消融实验（Ablation Study）证实了其在微小病灶特征捕捉上的显著增益。
- **打破黑盒与多维临床压测**：集成 3D Grad-CAM 逆向投影技术，精准定位良性“中心钙化”与恶性“边缘浸润”特征；并执行了极其严苛的临床亚组分析（Clinical Subgroups）与零样本泛化测试（LUNA16 Generalization）。
- **循证医学应用落地**：基于 Youden Index 计算出最佳临床防漏诊阈值，融合模型集成策略（Ensemble），最终交付高可用性的 Streamlit 靶向药辅助决策 Web 平台。
- **机器人穿刺辅助原型**：从 CT 物理坐标生成肋骨 AABB 避障约束，规划直线路径，并通过 CoppeliaSim 的 UR5 + simIK + Python ZeroMQ Remote API 执行可检查的仿真轨迹。旧的 Matplotlib 图/GIF 只用于概念展示，不代表关节级机器人仿真。

---

## 2. 核心技术栈 (Tech Stack)

-   **核心驱动语言**: `Python 3.9+`
-   **医疗影像解析底座**:
    *   `SimpleITK`: 支撑三维空间插值、形态学滤波网络及全维度矩阵仿射变换。
    *   `pydicom`: 极速剥离 DICOM 头部信息并提取脱敏临床元数据。
-   **科学计算与特征工程聚合**:
    *   `pandas`: 构建临床宏观账本、执行高维数据聚合与阵营级联筛选。
    *   `numpy`: 驱动三维空间距离运算、HU 标量截断及安全掩膜填充。
-   **深度学习框架**: `PyTorch`, `PyTorch Lightning`
-   **机器人运动学与空间几何**: 基于标准 DH 参数构建的 6-DOF 运动学求解器、Ray-AABB 射线碰撞检测（Slab Method）
-   **多维可视化与动态交互**: `Matplotlib 3D`, `Streamlit`, `Pillow (GIF 动图渲染)`

---

## 3. 项目标准目录结构 (Project Directory Structure)
本仓库的代码架构严格遵循高内聚、低耦合的机器学习工程规范：

```text
Lung-Nodule/
├── data/
│   ├── processed/                       # 数据资产核心库
│   │   ├── all_roi_cubes/               # LIDC 32x32x32 标准化结节张量
│   │   ├── luna16_roi_cubes/            # LUNA16 跨域泛化测试张量
│   │   ├── preprocessed_cubes/          # LUNA16 全肺闭运算预处理矩阵
│   │   ├── cleaned_nodules_labels.csv   # 黄金结节清洗清单 (Ground Truth)
│   │   ├── final_split_train_manifest.csv # 7:1:2 患者级防泄露切分索引
│   │   └── *manifest.csv                # 各阶段资产总账本
│   └── demo_nodules/                    # 手术环境描述、规划轨迹与三维仿真导出物
│
├── models/                              # 模型权重与评估产物
│   ├── checkpoints_*/                   # 细分模型权重库 (2d, 3d, vnet, attention, densenet, ablation)
│   ├── evaluation_results/              # 各类 ROC 曲线、混淆矩阵、统计分析图表
│   └── logs/                            # TensorBoard 训练日志
│
├── src/                                 # 核心闭环源码
│   ├── 预处理与数据管道 (Data Pipeline)
│   │   ├── preprocess.py / explore_data.py / extract_roi.py
│   │   ├── extract_luna16.py            # LUNA16 盲测集独立提取器
│   │   ├── split_dataset.py             # 患者级防穿越数据切分
│   │   └── dataset_2D.py / dataset_3D.py # PyTorch Dataset 加载器
│   │
│   ├── 核心网络架构 (Network Architectures)
│   │   └── model_*.py                   # 涵盖 2D, 3D, DenseNet, VNet 等骨干网络定义
│   │
│   ├── 训练引擎 (Training Modules)
│   │   └── train_*.py                   # 针对不同架构的独立训练脚本 (含 Attention 与 Ablation)
│   │
│   ├── 评估与可解释性 (Evaluation & Interpretability)
│   │   ├── evaluate_*.py                # 多维评估 (2D/3D对比, Subgroups, 泛化测试, Ensemble, 临床阈值)
│   │   └── generate_3D_gradcam.py       # 3D 梯度类激活热力图反演
│   │
│   ├── 手术环境建模与轨迹规划 (Environment & Trajectory Planning)
│   │   ├── environment_modeling.py      # CT 骨骼与胸壁包围盒 (AABB) 提取
│   │   └── trajectory_planner.py        # 基于射线-AABB 碰撞检测的最优路径规划
│   │
│   ├── 机器人运动控制与多模态仿真 (Robotics & Simulation)
│   │   ├── robot_kinematics.py          # UR5 机械臂 DH 建模与逆运动学数值求解
│   │   ├── robot_geometry.py             # CT/机器人坐标变换、单位和姿态工具
│   │   ├── main_pipeline.py             # 端到端 AI+机器人穿刺总控主流程
│   │   ├── coppeliasim_remote.py        # CoppeliaSim ZeroMQ + simIK 执行器
│   │   ├── visualize_simulation.py      # 3D 静态手术场景仿真与高清图导出
│   │   └── simulate_dynamic_puncture.py # 动态进针穿刺模拟与演示 GIF 导出
│   │
│   └── 部署与交互 (Deployment)
│       ├── inference_api.py             # 工业级推理张量解析接口
│       └── app.py                       # 👨‍⚕️ Streamlit 临床决策 Web 原型
│
└── requirements.txt                     # 环境依赖清单
```

---

## 4. 全链路运行指南 (Execution Guide)

> **推荐入口**：机器人部分已经封装为病例配置驱动脚本。新病例优先阅读
> [`PIPELINE_GUIDE.md`](PIPELINE_GUIDE.md)，并使用
> `python src/run_case_pipeline.py --config configs/cases_0036_0037_0039.json --case-id <CASE_ID>`。
> 下文保留底层命令，便于调试单个阶段。
为确保整个数据流动管道平滑闭环且表头无污染，请严格按照以下一键通关顺序执行：

**阶段一**：数据清洗与物理空间重构
请依次执行以下脚本，完成异构数据的提纯、聚类与 3D ROI 提取：
```text
python src/preprocess.py
python src/explore_data.py
python src/extract_roi.py
python src/extract_luna16.py
python src/split_dataset.py
```

**阶段二**：算法矩阵训练与消融实验
项目支持多维度网络架构的训练对比，可根据研究目的运行对应脚本：
* 基线模型训练对比
```text
python src/train_2D.py
python src/train_3D.py
python src/train_3D_vnet.py
```

* 注意力机制与消融实验
```text
python src/train_3D_attention.py
python src/train_3D_ablation.py
```

**阶段三**：多维临床压测与可解释性反演
模型训练完成后，执行全面的科学评估与视觉映射：
* 基础性能与盲测泛化评估
```text
python src/evaluate_3D_models.py
python src/evaluate_luna16_generalization.py
```

* 临床亚组分析与模型融合
```text
python src/evaluate_clinical_subgroups.py
python src/evaluate_ensemble.py
```

* 打破黑盒：生成 3D 浸润热力图
```text
python src/generate_3D_gradcam.py
```

**阶段四**：寻找最佳临床工作点与 Web 部署
计算最优敏感/特异性平衡点，并启动交互式诊疗系统：
* 锁定 Youden Index 最佳阈值
```text
python src/evaluate_clinical_value.py
```

* 启动 Web 前端交互系统
```text
streamlit run src/app.py
```

**阶段五**：机器人规划与 CoppeliaSim 验证
下面的顺序是机器人部分的实际依赖链。不要把旧 GIF 当作 CoppeliaSim 验证结果：
* 提取解剖避障约束
```text
PYTHONPATH="/PATH/TO/pydicom/site-packages" python src/environment_modeling.py \
  --ct-path "/ABS/PATH/TO/EXACT/SERIES/FOLDER" \
  --series-instance-uid "SERIES_INSTANCE_UID_FROM_DICOM" \
  --case-id "LIDC-IDRI-0037" \
  --target-voxel 407 289 79 \
  --bone-hu-threshold 500 \
  --min-volume-voxels 15 \
  --segment-slices 1 \
  --output data/demo_nodules/rib_obstacles_0037_segmented.json \
  --preview-output data/demo_nodules/rib_obstacles_0037_segmented_preview.png
```

`--target-voxel` is `[x, y, z]` in the selected CT series, not a physical
millimetre coordinate. It must be the same ROI centre used to create the
checkpoint input. The extractor refuses to run when `--series-instance-uid`
does not match the DICOM headers. The generated JSON is a conservative,
segmented obstacle approximation for simulation; inspect the preview PNG before
using it for trajectory planning. It is not a clinical rib segmentation.

For a new case, do not reuse the 0037 lateral-window values blindly. Recompute
the target voxel and use a symmetric local window when the lesion side or CT
orientation differs, for example `--x-min-offset-mm -140
--x-max-offset-mm 140`. The planning code can consume the resulting JSON without
case-specific changes.

The planner's `multi_direction` mode samples radial entry directions and
distances in the CT physical frame, checks every segment against the obstacle
AABBs, rejects targets that fall inside an obstacle, and then performs
analytical UR5 IK. This makes the software reusable across cases, but the data
preparation is still case-specific: every new case needs its own exact
SeriesInstanceUID, ROI target voxel/physical coordinate, obstacle JSON, QC
preview, and patient-to-robot transform. The `.npy` ROI cube alone does not
contain enough metadata to recover the DICOM target coordinate.

* 运行 AI 推理、避障规划和分析 IK，并写出 CoppeliaSim 输入契约
```text
python src/main_pipeline.py --checkpoint "/Users/vivi/Documents/BN5211/Final Project/训练好的/3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt" --roi "/Users/vivi/Documents/BN5211/Final Project/训练好的/LIDC-IDRI-0037_IL057_12.npy" --target-mm 118.275 20.625 -163.50 --obstacles data/demo_nodules/rib_obstacles_0037_segmented.json --planning-mode multi_direction --entry-distances-mm 60 80 100 --angular-step-deg 15 --output-plan data/demo_nodules/planned_trajectory_0037.json --confirm-coincident-demo-frame --confirm-target-roi-match --force-robot-demo
```

* 启动 CoppeliaSim 场景并由 simIK 驱动 UR5。场景对象名称、关节顺序和碰撞集合必须先按第 6 节创建。
```text
PYTHONPATH="/PATH/TO/coppeliaSim.app/Contents/Resources/programming/zmqRemoteApi/clients/python/src" python src/coppeliasim_remote.py --plan data/demo_nodules/planned_trajectory_0037.json --scene scenes/robotic_lung_biopsy.ttt --load-scene --robot /UR5 --ik-tip /UR5/ikTip --ik-target /UR5/ikTarget --robot-collision-collection /robotCollision --environment-collision-collection /patientCollision --kinematic-only --output data/demo_nodules/coppeliasim_run_0037.json
```

For the current 037 segmented case, use
`scenes/robotic_lung_biopsy_0037_segmented.ttt` and
`data/demo_nodules/rib_obstacles_0037_segmented.json`. The kinematic-only
executor starts the stock UR5 from `[0, -pi/2, 0, -pi/2, 0, 0]` radians, then
commands the pre-approach point before the approach-to-entry and insertion
segments. Override the six start angles with six repeated
`--initial-joint-angle-rad` options when a different scene requires it.

On macOS, start CoppeliaSim directly so the ZeroMQ server is available:

```text
/PATH/TO/coppeliaSim.app/Contents/MacOS/coppeliaSim -GzmqRemoteApi.rpcPort=23000
```

Run the Python client in the same local permission context as CoppeliaSim. A
startup warning about the optional `pythonLauncher.py` wrapper missing `zmq`
does not affect this bundled ZeroMQ Remote API client, which uses `pyzmq` and
`cbor` from the selected Python environment.

The repository includes a reproducible integration scene at
`scenes/robotic_lung_biopsy.ttt`. If it needs to be regenerated after changing
the UR5 model or obstacle subset:

```text
PYTHONPATH="/PATH/TO/coppeliaSim.app/Contents/Resources/programming/zmqRemoteApi/clients/python/src" python src/create_coppeliasim_scene.py --output scenes/robotic_lung_biopsy.ttt --obstacles data/demo_nodules/rib_obstacles_0037.json --max-obstacles 1
```

`--kinematic-only` is required for the stock UR5 model used here: it keeps the
scene stopped, switches the six joints to kinematic mode, and prevents the
model's dynamic controller from overwriting simIK joint positions. The
`robotCollision` and `patientCollision` collections are recreated at runtime
because CoppeliaSim collections are not persisted in the `.ttt` scene.
Add `--frame-delay 0.08` to the remote command when you want to watch the
actual CoppeliaSim motion frame by frame in the GUI. Without this option the
same 100 frames are executed too quickly to observe.

* 旧的 Matplotlib 概念图（可选，不是机器人验证）
```text
python src/simulate_dynamic_puncture.py
```

---

## 5. 快速开始与环境配置 (Quick Start)

**1**. 创建并激活纯净虚拟环境
```text
conda create -n lung_nodule python=3.9 -y
conda activate lung_nodule
```

**2**. 安装核心依赖
```text
pip install -r requirements.txt
```

---

## 6. 机器人 P0 接口与人工验收

### 6.1 坐标和单位契约

- CT 结节、入针点、肋骨 AABB 使用 **SimpleITK physical coordinates，单位 mm**。
- `main_pipeline.py` 输出的 `patient_to_robot_base` 是从 CT 患者坐标到 UR5 base 坐标的 4x4 刚体变换，长度仍是 mm。
- CoppeliaSim 内部使用 m/rad；`coppeliasim_remote.py` 只在 API 边界把 mm 转为 m。
- `/UR5/ikTip` 必须放在物理针尖处，而不是只放在法兰处；`/UR5/ikTarget` 是它的目标 dummy。
- `--confirm-coincident-demo-frame` 只允许一个明确的教学演示：它表示场景中 CT frame 与 UR5 base frame 被人为摆成一致。真实实验必须提供实际标定变换，不能使用 identity。
- `--confirm-target-roi-match` 是另一道独立保护：操作者必须确认 ROI 文件、target 坐标和障碍物 JSON 属于同一病例/同一坐标系。文件名相似不能替代这个确认。
- 轨迹执行分为三段：当前针尖到体外预接近点、预接近点到皮肤入口、入口到靶点的 insertion。预接近点默认位于入口点外侧 30 mm，可用 `--approach-distance-mm` 调整。
- 场景启动时 UR5 的初始针尖位姿必须是安全、可达且位于患者模型外；脚本会从该实际位姿开始记录 `initial_tcp_to_approach`，不会把任意初始姿态伪装成已经到达入口点。

### 6.2 需要本地修改的路径

`src/main_pipeline.py` 的默认 checkpoint/ROI 是当前电脑的路径：

```text
/Users/vivi/Documents/BN5211/Final Project/训练好的/3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt
/Users/vivi/Documents/BN5211/Final Project/训练好的/LIDC-IDRI-0037_IL057_12.npy
```

换电脑时用命令行 `--checkpoint` 和 `--roi` 覆盖。还需要提供：

- 与 ROI/target 同一病例、同一坐标系的 CT DICOM 序列；
- 由 `environment_modeling.py` 生成的 `rib_obstacles_0037.json`；
- CoppeliaSim `.ttt` 场景；
- CT 患者坐标到 UR5 base 的标定矩阵，或仅用于教学的 coincident-frame 场景；
- 场景中的 UR5 关节路径、IK tip/target dummy 和机器人/患者碰撞集合。

### 6.3 人工验证顺序

1. 创建 Python 3.9 环境并安装 `requirements.txt`。机器人推理入口直接加载已有 CBAM checkpoint，不需要导入 PyTorch Lightning；训练脚本仍需要 `pytorch-lightning`。先执行 `python -c "import torch; print(torch.__version__)"`。
2. 确认 ROI 是 `(32, 32, 32)`，并把 CBAM checkpoint 路径指向 `3d_resnet_cbam-epoch=29-val_acc=0.84.ckpt`。不要用 `3d_resnet18` checkpoint 加载 CBAM 模型。
3. 如果有原始 CT，先运行 `python src/environment_modeling.py`，检查输出 AABB JSON 与患者坐标单位；若没有 CT，不能伪造障碍物 JSON。
4. 运行第 5 阶段的 `main_pipeline.py` 命令。若有真实标定矩阵，增加 `--patient-to-robot-transform /ABS/PATH/ct_to_ur5_mm.json`；只有教学场景才增加 `--confirm-coincident-demo-frame`。同时人工核对 ROI、target、障碍物 JSON 属于同一病例后增加 `--confirm-target-roi-match`。检查输出 JSON 中 `decision=robot_plan_ready`、`best_trajectory`、患者/机器人两套坐标、`approach_point_*` 以及 `analytical_ik`。如果 AI 概率低于阈值，教学演示才使用 `--force-robot-demo`。
5. 在 CoppeliaSim 打开 `.ttt` 场景，确认 UR5 有 6 个关节，`ikTip` 位于针尖，`ikTarget` 位于同一世界坐标体系，并能看到患者/肋骨碰撞集合。当前场景默认使用 `[0, -pi/2, 0, -pi/2, 0, 0]` radians 作为安全起始姿态；执行器会先把预接近点作为第一条 IK 目标。
6. 启用 CoppeliaSim ZeroMQ Remote API 服务（默认端口 `23000`），执行 `coppeliasim_remote.py` 命令。脚本会读取初始针尖位置，逐段调用 simIK，记录 `phase_sequence`，读取实际 tip 位姿并调用 `sim.checkCollision`。
7. 只有当输出 `collision_free=true`、`max_tip_position_error_mm` 在项目报告设定阈值内、`collision_frames` 为空且 `phase_sequence` 包含 insertion 时，才记录仿真通过。`coppeliasim_run_0037.json` 是实际远程运行结果；规划 JSON 本身不是运行结果。

### 6.4 当前明确未完成/缺失

仓库现在已有软件集成用 `.ttt` 场景、UR5、针具、IK dummies、运行时碰撞集合重建逻辑和多方向直线规划器。规划器仍是仿真级几何规划：它没有自动验证真实皮肤表面，也没有呼吸运动、软组织、血管或临床器械约束。真实患者到 UR5 的实测 4x4 标定矩阵仍缺失；没有标定时只能使用明确标注的 coincident-frame 教学演示。原始 CT 序列可以不随仓库提交，但每个病例生成或复核障碍物时必须在本地提供与 ROI/target 同病例的 DICOM 序列。

本仓库还提供一个不依赖 CT 规划结果的 CoppeliaSim 软件冒烟测试：

```text
PYTHONPATH="/PATH/TO/coppeliaSim.app/Contents/Resources/programming/zmqRemoteApi/clients/python/src" python src/coppeliasim_remote.py --plan data/demo_nodules/coppeliasim_smoke_plan.json --scene scenes/robotic_lung_biopsy.ttt --load-scene --robot /UR5 --ik-tip /UR5/ikTip --ik-target /UR5/ikTarget --robot-collision-collection /robotCollision --environment-collision-collection /patientCollision --steps 24 --kinematic-only --output data/demo_nodules/coppeliasim_smoke_run.json
```

通过标准：输出 JSON 中 `collision_free=true`、`collision_frames=[]`、
`phase_sequence` 包含 `approach`、`approach_to_entry`、`insertion`，且
`max_tip_position_error_mm` 小于项目报告设定阈值。该 smoke plan 只验证
软件连接和 simIK，不是患者穿刺计划。

---

## 7. 免责声明 (Disclaimer)
本项目模型及 Web 系统输出的恶性概率与风险分层结果，仅供生物医学工程领域的学术研究、算法验证及演示使用。在任何情况下，均不可替代专业注册放射科医师与肿瘤科医师在真实医疗环境中的最终确诊意见。规划轨迹与机械臂控制仿真代码亦不可直接用于真实人体手术操作。
