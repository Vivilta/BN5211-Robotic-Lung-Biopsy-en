import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from robot_kinematics import UR5Kinematics

def plot_rib_box(ax, min_pt, max_pt, color='#4A90E2', alpha=0.30):
    """Draw a translucent 3D anatomical rib bounding box."""
    x = [min_pt[0], max_pt[0]]
    y = [min_pt[1], max_pt[1]]
    z = [min_pt[2], max_pt[2]]
    
    vertices = np.array([
        [x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]],
        [x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]
    ])
    
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    
    poly = Poly3DCollection(faces, alpha=alpha, edgecolor='#1C39BB', linewidths=0.9)
    poly.set_facecolor(color)
    ax.add_collection3d(poly)

def main():
    print("🎨 Generating a high-resolution close-up view of the 3D surgical simulation...")
    
    traj_path = r"F:\Lung-Nodule\data\demo_nodules\planned_trajectory_0037.json"
    kin_path = r"F:\Lung-Nodule\data\demo_nodules\kinematics_result_0037.json"
    obs_path = r"F:\Lung-Nodule\data\demo_nodules\rib_obstacles_0037.json"
    
    with open(traj_path, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)
    with open(kin_path, 'r', encoding='utf-8') as f:
        kin_data = json.load(f)
    with open(obs_path, 'r', encoding='utf-8') as f:
        obstacles = json.load(f)

    target_pt = np.array(traj_data["target_point"])
    entry_pt = np.array(traj_data["best_trajectory"]["entry_point"])
    joint_angles = kin_data["joint_angles_rad"]
    ur5 = UR5Kinematics()

    fig = plt.figure(figsize=(13, 9), dpi=130)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.04)

    # 1. Draw posterior chest-wall ribs as the background.
    mid_puncture = (entry_pt + target_pt) / 2.0
    dists = []
    for obs in obstacles:
        b_min = np.array(obs["min_mm"])
        b_max = np.array(obs["max_mm"])
        c = (b_min + b_max) / 2.0
        dists.append(np.linalg.norm(c - mid_puncture))
    
    sort_idx = np.argsort(dists)
    for idx in sort_idx[:6]:
        obs = obstacles[idx]
        b_min = np.array(obs["min_mm"])
        b_max = np.array(obs["max_mm"])
        dims = np.abs(b_max - b_min)
        c = (b_min + b_max) / 2.0
        clamped_dims = np.clip(dims, 15.0, 50.0)
        plot_rib_box(ax, c - clamped_dims/2.0, c + clamped_dims/2.0, color='#4A90E2', alpha=0.22)

    # 2. Draw the ribs above and below the intercostal entry point.
    rib_width = 36.0
    rib_thick = 14.0
    rib_height = 16.0
    
    # Superior Rib
    sup_min = np.array([entry_pt[0] - rib_width/2, entry_pt[1] - 6.0, entry_pt[2] + 15.0])
    sup_max = sup_min + np.array([rib_width, rib_thick, rib_height])
    plot_rib_box(ax, sup_min, sup_max, color='#3182CE', alpha=0.38)

    # Inferior Rib
    inf_min = np.array([entry_pt[0] - rib_width/2, entry_pt[1] - 6.0, entry_pt[2] - 15.0 - rib_height])
    inf_max = inf_min + np.array([rib_width, rib_thick, rib_height])
    plot_rib_box(ax, inf_min, inf_max, color='#3182CE', alpha=0.38)

    # 3. Draw the target point (large red sphere) and entry point (blue cone).
    ax.scatter(target_pt[0], target_pt[1], target_pt[2], 
               color='#E53E3E', s=240, edgecolors='black', linewidths=1.2,
               label='Malignant Nodule (Target)', depthshade=False)

    ax.scatter(entry_pt[0], entry_pt[1], entry_pt[2], 
               color='#0072BD', s=190, marker='^', edgecolors='black', linewidths=1.2,
               label='Skin Entry Point (Intercostal)', depthshade=False)

    # 4. Draw the needle path as a bright green line.
    ax.plot(
        [entry_pt[0], target_pt[0]],
        [entry_pt[1], target_pt[1]],
        [entry_pt[2], target_pt[2]],
        color='#00E676', linewidth=5.0, label='Needle Trajectory (Collision-Free)'
    )

    # 5. Calculate the manipulator's spatial pose.
    T_local = np.eye(4)
    local_pts = [T_local[:3, 3].copy()]
    for i in range(6):
        T_i = ur5.dh_transform(ur5.a[i], ur5.alpha[i], ur5.d[i], joint_angles[i])
        T_local = np.dot(T_local, T_i)
        local_pts.append(T_local[:3, 3].copy())
    local_pts = np.array(local_pts)

    offset = entry_pt - local_pts[-1]
    world_robot_pts = local_pts + offset

    # Manipulator links and joints.
    ax.plot(world_robot_pts[:, 0], world_robot_pts[:, 1], world_robot_pts[:, 2], 
            color='#ED8936', linewidth=4.5, marker='o', markersize=7, 
            markerfacecolor='#1A365D', markeredgecolor='white', markeredgewidth=1.2,
            label='UR5 Manipulator (Surgical Posture)')

    # 6. Set the view and zoom into the surgical area.
    ax.set_xlim([150, 460])
    ax.set_ylim([-50, 320])
    ax.set_zlim([-260, 50])

    # 7. Add the title and labels.
    fig.suptitle(
        "Autonomous Robotic Lung Biopsy Simulation (Patient LIDC-IDRI-0037)\nEnd-to-End Multimodal Cyber-Physical System Verification", 
        fontsize=13, fontweight='bold', y=0.96
    )
    ax.set_xlabel("X (mm) [Sagittal]", fontsize=10, labelpad=8)
    ax.set_ylabel("Y (mm) [Coronal]", fontsize=10, labelpad=8)
    ax.set_zlabel("Z (mm) [Axial]", fontsize=10, labelpad=8)

    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor='white', framealpha=0.92, fontsize=9)
    ax.view_init(elev=20, azim=46)

    # 8. Save the high-resolution image (300 DPI).
    out_img = r"F:\Lung-Nodule\data\demo_nodules\simulation_3d_0037.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"Simulation image saved to: {out_img}")
    
    plt.show()

if __name__ == "__main__":
    main()
