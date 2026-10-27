import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

from robot_kinematics import UR5Kinematics

def plot_rib_box(ax, min_pt, max_pt, color='#4A90E2', alpha=0.30):
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
    print("🎬 Starting the dynamic surgical puncture animation generator...")
    
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

    fig = plt.figure(figsize=(13, 9), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.04)

    # 1. Static background: posterior chest-wall ribs.
    mid_puncture = (entry_pt + target_pt) / 2.0
    dists = [np.linalg.norm((np.array(o["min_mm"]) + np.array(o["max_mm"]))/2.0 - mid_puncture) for o in obstacles]
    sort_idx = np.argsort(dists)
    for idx in sort_idx[:6]:
        obs = obstacles[idx]
        b_min, b_max = np.array(obs["min_mm"]), np.array(obs["max_mm"])
        c = (b_min + b_max) / 2.0
        dims = np.clip(np.abs(b_max - b_min), 15.0, 50.0)
        plot_rib_box(ax, c - dims/2.0, c + dims/2.0, color='#4A90E2', alpha=0.22)

    # 2. Static background: anterior ribs around the intercostal space.
    rib_w, rib_t, rib_h = 36.0, 14.0, 16.0
    sup_min = np.array([entry_pt[0] - rib_w/2, entry_pt[1] - 6.0, entry_pt[2] + 15.0])
    plot_rib_box(ax, sup_min, sup_min + np.array([rib_w, rib_t, rib_h]), color='#3182CE', alpha=0.38)
    inf_min = np.array([entry_pt[0] - rib_w/2, entry_pt[1] - 6.0, entry_pt[2] - 15.0 - rib_h])
    plot_rib_box(ax, inf_min, inf_min + np.array([rib_w, rib_t, rib_h]), color='#3182CE', alpha=0.38)

    # 3. Static target and entry points.
    ax.scatter(target_pt[0], target_pt[1], target_pt[2], color='#E53E3E', s=240, edgecolors='black', label='Malignant Nodule (Target)', depthshade=False)
    ax.scatter(entry_pt[0], entry_pt[1], entry_pt[2], color='#0072BD', s=190, marker='^', edgecolors='black', label='Skin Entry Point', depthshade=False)

    # 4. Calculate the manipulator's reference skeleton.
    T_local = np.eye(4)
    local_pts = [T_local[:3, 3].copy()]
    for i in range(6):
        T_i = ur5.dh_transform(ur5.a[i], ur5.alpha[i], ur5.d[i], joint_angles[i])
        T_local = np.dot(T_local, T_i)
        local_pts.append(T_local[:3, 3].copy())
    local_pts = np.array(local_pts)
    offset = entry_pt - local_pts[-1]
    static_robot_pts = local_pts + offset

    # Initialize dynamic line elements.
    needle_line, = ax.plot([], [], [], color='#00E676', linewidth=5.0, label='Needle Trajectory (Advancing)')
    robot_line, = ax.plot([], [], [], color='#ED8936', linewidth=4.5, marker='o', markersize=7, 
                          markerfacecolor='#1A365D', markeredgecolor='white', label='UR5 Manipulator')
    status_text = ax.text2D(0.02, 0.78, "", transform=ax.transAxes, fontsize=11, fontweight='bold', color='#D9381E')

    # Set the field of view.
    ax.set_xlim([150, 460])
    ax.set_ylim([-50, 320])
    ax.set_zlim([-260, 50])
    fig.suptitle("CONCEPT-ONLY Animation (not a CoppeliaSim robot simulation)\nRigidly translated arm illustration; no joint motion or collision validation",
                 fontsize=13, fontweight='bold', y=0.96)
    ax.set_xlabel("X (mm)", fontsize=10)
    ax.set_ylabel("Y (mm)", fontsize=10)
    ax.set_zlabel("Z (mm)", fontsize=10)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, facecolor='white', framealpha=0.92, fontsize=9)
    ax.view_init(elev=20, azim=46)

    # Number of animation frames.
    total_frames = 60
    
    def update(frame):
        # Insertion progress alpha ranges from 0.0 (entry point) to 1.0 (nodule target).
        progress = frame / float(total_frames)
        current_tip = entry_pt + progress * (target_pt - entry_pt)
        depth = progress * np.linalg.norm(target_pt - entry_pt)

        # Update the needle trajectory.
        needle_line.set_data([entry_pt[0], current_tip[0]], [entry_pt[1], current_tip[1]])
        needle_line.set_3d_properties([entry_pt[2], current_tip[2]])

        # Advance the manipulator illustration with the needle holder.
        current_robot_pts = static_robot_pts + (current_tip - entry_pt)
        robot_line.set_data(current_robot_pts[:, 0], current_robot_pts[:, 1])
        robot_line.set_3d_properties(current_robot_pts[:, 2])

        status_text.set_text(f"Illustration only\nInsertion visualization: {progress*100:.1f}%\nNo joint-level simulation or collision validation")
        return needle_line, robot_line, status_text

    print(">> Rendering the animation and exporting a GIF. Please wait...")
    anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
    
    # Export the GIF animation.
    out_gif = r"F:\Lung-Nodule\data\demo_nodules\dynamic_biopsy_simulation.gif"
    try:
        anim.save(out_gif, writer='pillow', fps=20)
        print(f"🎉 Dynamic surgery GIF generated: {out_gif}")
    except Exception as e:
        print(f"⚠️ GIF save warning: {e}")

    plt.show()

if __name__ == "__main__":
    main()
