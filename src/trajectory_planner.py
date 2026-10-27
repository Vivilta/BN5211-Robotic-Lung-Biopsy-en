import os
import json
import numpy as np

def ray_intersects_aabb(p_start, p_end, aabb_min, aabb_max):
    """
    Classic 3D ray-versus-axis-aligned bounding box (AABB) collision test
    using the slab method. Check whether segment [p_start, p_end] intersects
    the box bounded by [aabb_min, aabb_max].
    """
    d = p_end - p_start
    t_min = 0.0
    t_max = 1.0

    for i in range(3):
        if abs(d[i]) < 1e-6:
            # The ray is parallel to this axis.
            if p_start[i] < aabb_min[i] or p_start[i] > aabb_max[i]:
                return False
        else:
            t1 = (aabb_min[i] - p_start[i]) / d[i]
            t2 = (aabb_max[i] - p_start[i]) / d[i]
            t_near = min(t1, t2)
            t_far = max(t1, t2)

            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)

            if t_min > t_max:
                return False

    return True

class BiopsyTrajectoryPlanner:
    """
    Percutaneous lung biopsy trajectory planner for entry-point search
    and straight-line collision avoidance.
    """
    def __init__(self, obstacles_json_path, safety_margin_mm=2.0, needle_radius_mm=0.0):
        if not os.path.isfile(obstacles_json_path):
            raise FileNotFoundError(f"Obstacle JSON not found: {obstacles_json_path}")
        with open(obstacles_json_path, 'r', encoding='utf-8') as f:
            self.obstacles = json.load(f)
        if not isinstance(self.obstacles, list):
            raise ValueError("Obstacle JSON must contain a list of AABB objects")
        self.safety_margin_mm = float(safety_margin_mm)
        self.needle_radius_mm = float(needle_radius_mm)
        if self.safety_margin_mm < 0 or self.needle_radius_mm < 0:
            raise ValueError("Safety margin and needle radius must be non-negative")
        self.obstacle_inflation_mm = self.safety_margin_mm + self.needle_radius_mm
        print(f"✅ Loaded {len(self.obstacles)} rib obstacle bounding boxes.")

    def check_collision(self, entry_pt, target_pt):
        """
        Check whether the line segment from entry_pt to target_pt intersects a rib.
        """
        entry_pt = np.array(entry_pt)
        target_pt = np.array(target_pt)

        for obs in self.obstacles:
            box_min = np.asarray(obs['min_mm'], dtype=float)
            box_max = np.asarray(obs['max_mm'], dtype=float)

            # Apply the configured planning margin in patient CT millimetres.
            safety_margin = self.obstacle_inflation_mm
            box_min -= safety_margin
            box_max += safety_margin

            if ray_intersects_aabb(entry_pt, target_pt, box_min, box_max):
                return True, obs['id']  # Return the ID of the intersected rib.
        return False, None

    def target_clearance_report(self, target_pt):
        """Report obstacle AABBs that contain or approach the target point."""
        target_pt = np.asarray(target_pt, dtype=float)
        if target_pt.shape != (3,):
            raise ValueError("target_pt must contain exactly three coordinates")

        report = []
        for obs in self.obstacles:
            box_min = np.asarray(obs["min_mm"], dtype=float) - self.obstacle_inflation_mm
            box_max = np.asarray(obs["max_mm"], dtype=float) + self.obstacle_inflation_mm
            closest = np.maximum(box_min, np.minimum(target_pt, box_max))
            distance = float(np.linalg.norm(target_pt - closest))
            inside = bool(np.all(target_pt >= box_min) and np.all(target_pt <= box_max))
            report.append(
                {
                    "obstacle_id": obs.get("id", "unnamed"),
                    "distance_mm": distance,
                    "contains_target": inside,
                }
            )
        return sorted(report, key=lambda item: item["distance_mm"])

    @staticmethod
    def _sample_unit_directions(angular_step_deg):
        """Deterministically sample outward directions on a sphere."""
        if angular_step_deg <= 0 or angular_step_deg > 90:
            raise ValueError("angular_step_deg must be in (0, 90]")
        directions = []
        elevations = np.arange(-90.0, 90.0 + angular_step_deg * 0.5, angular_step_deg)
        azimuths = np.arange(0.0, 360.0, angular_step_deg)
        for elevation_deg in elevations:
            elevation = np.deg2rad(elevation_deg)
            cos_elevation = np.cos(elevation)
            for azimuth_deg in azimuths:
                azimuth = np.deg2rad(azimuth_deg)
                direction = np.array(
                    [
                        cos_elevation * np.cos(azimuth),
                        cos_elevation * np.sin(azimuth),
                        np.sin(elevation),
                    ],
                    dtype=float,
                )
                if not any(np.allclose(direction, existing) for existing in directions):
                    directions.append(direction)
        return directions

    def plan_multidirectional(
        self,
        target_pt,
        entry_distances_mm=(60.0, 80.0, 100.0),
        angular_step_deg=15.0,
        preferred_direction=None,
        max_direction_deviation_deg=None,
        approach_distance_mm=0.0,
        needle_length_mm=0.0,
    ):
        """Search straight-line entry paths from multiple patient-frame directions.

        This is a geometry-only search. Without a body/skin surface model, the
        sampled entry point is a radial point outside the target, not a verified
        anatomical skin point. A future body-mask adapter can replace the radial
        candidates without changing collision checking or scoring.
        """
        target_pt = np.asarray(target_pt, dtype=float)
        if target_pt.shape != (3,) or not np.isfinite(target_pt).all():
            raise ValueError("target_pt must contain three finite coordinates")
        distances = np.asarray(tuple(entry_distances_mm), dtype=float)
        if distances.size == 0 or np.any(distances <= 0):
            raise ValueError("entry_distances_mm must contain positive values")
        if approach_distance_mm < 0 or needle_length_mm < 0:
            raise ValueError("approach_distance_mm and needle_length_mm must be non-negative")

        preferred = None
        if preferred_direction is not None:
            preferred = np.asarray(preferred_direction, dtype=float)
            if preferred.shape != (3,) or not np.isfinite(preferred).all():
                raise ValueError("preferred_direction must contain three finite values")
            norm = np.linalg.norm(preferred)
            if norm == 0:
                raise ValueError("preferred_direction must be non-zero")
            preferred = preferred / norm
        if max_direction_deviation_deg is not None and not 0 < max_direction_deviation_deg <= 180:
            raise ValueError("max_direction_deviation_deg must be in (0, 180]")

        target_clearance = self.target_clearance_report(target_pt)
        if target_clearance and target_clearance[0]["contains_target"]:
            print(
                "❌ Target lies inside a safety-margin-expanded obstacle AABB; refusing to generate a path: "
                f"{target_clearance[0]['obstacle_id']}"
            )
            return None

        directions = self._sample_unit_directions(angular_step_deg)
        candidates = []
        total_tested = 0
        for outward in directions:
            if preferred is not None:
                deviation = np.rad2deg(np.arccos(np.clip(np.dot(outward, preferred), -1.0, 1.0)))
                if max_direction_deviation_deg is not None and deviation > max_direction_deviation_deg:
                    continue
            else:
                deviation = 0.0

            for distance in distances:
                total_tested += 1
                entry_pt = target_pt + outward * distance
                approach_pt = entry_pt + outward * float(approach_distance_mm)
                needle_sweep_start = approach_pt + outward * float(needle_length_mm)
                collision, hit_id = self.check_collision(needle_sweep_start, target_pt)
                if collision:
                    continue
                candidates.append(
                    {
                        "entry_point": entry_pt.tolist(),
                        "approach_point": approach_pt.tolist(),
                        "needle_sweep_start": needle_sweep_start.tolist(),
                        "path_length": float(distance),
                        "vector": (-outward).tolist(),
                        "entry_direction_outward": outward.tolist(),
                        "planning_mode": "multi_direction_radial",
                        "collision_free": True,
                        "first_collision_obstacle": hit_id,
                        "direction_deviation_deg": float(deviation),
                    }
                )

        print("\n================ Multi-direction collision-free path planning ================")
        print(f"Candidate paths evaluated: {total_tested}")
        print(f"Collision-free paths: {len(candidates)}")
        success_rate = (len(candidates) / total_tested) * 100 if total_tested else 0.0
        print(f"📈 Collision-free candidate rate: {success_rate:.2f}%")
        if not candidates:
            print("❌ No collision-free straight path found across the sampled directions.")
            return None

        # Prefer the shortest needle path; use angular deviation only as a
        # deterministic tie-breaker when a preferred clinical direction exists.
        best_path = min(
            candidates,
            key=lambda item: (item["path_length"], item["direction_deviation_deg"]),
        )
        print(f">> Best entry point: {[round(c, 2) for c in best_path['entry_point']]} mm")
        print(f">> Insertion depth: {best_path['path_length']:.2f} mm")
        print("========================================================\n")
        return best_path

    def plan_trajectory(
        self,
        target_pt,
        skin_y_coord=None,
        search_radius=40.0,
        step=5.0,
        approach_distance_mm=0.0,
        needle_length_mm=0.0,
    ):
        """
        Search for a safe entry point on a grid over the skin region in front
        of the target. target_pt is [x, y, z] in mm. skin_y_coord is the assumed
        Y coordinate of the anterior chest wall; if None, use 60 mm in front
        of the target.
        """
        target_pt = np.asarray(target_pt, dtype=float)
        if target_pt.shape != (3,):
            raise ValueError("target_pt must contain exactly three coordinates")
        if search_radius <= 0 or step <= 0:
            raise ValueError("search_radius and step must be positive")
        if approach_distance_mm < 0 or needle_length_mm < 0:
            raise ValueError("approach_distance_mm and needle_length_mm must be non-negative")
        
        # Approximate anterior chest-wall skin location: 50-70 mm in front of the target along Y.
        if skin_y_coord is None:
            skin_y_coord = target_pt[1] - 60.0

        candidates = []
        # Sample a circular grid region in the X-Z plane around the target.
        x_range = np.arange(target_pt[0] - search_radius, target_pt[0] + search_radius + 1e-3, step)
        z_range = np.arange(target_pt[2] - search_radius, target_pt[2] + search_radius + 1e-3, step)

        total_tested = 0
        collision_free_paths = []

        for x in x_range:
            for z in z_range:
                # Sample only within the search-radius circle.
                dist_xy = np.sqrt((x - target_pt[0])**2 + (z - target_pt[2])**2)
                if dist_xy > search_radius:
                    continue

                entry_pt = np.array([x, skin_y_coord, z])
                total_tested += 1
                path_length = np.linalg.norm(entry_pt - target_pt)
                if path_length <= 0:
                    continue
                direction = (target_pt - entry_pt) / path_length
                approach_pt = entry_pt - direction * float(approach_distance_mm)
                needle_sweep_start = approach_pt - direction * float(needle_length_mm)
                is_collision, hit_id = self.check_collision(needle_sweep_start, target_pt)

                if not is_collision:
                    collision_free_paths.append({
                        "entry_point": entry_pt.tolist(),
                        "approach_point": approach_pt.tolist(),
                        "needle_sweep_start": needle_sweep_start.tolist(),
                        "path_length": path_length,
                        "vector": ((target_pt - entry_pt) / path_length).tolist()
                    })

        # Calculate the proposal metric: trajectory feasibility rate.
        success_rate = (len(collision_free_paths) / total_tested) * 100 if total_tested > 0 else 0
        print("\n================ Collision-avoidance planning results ================")
        print(f"Entry points evaluated: {total_tested}")
        print(f"Safe, collision-free entry points: {len(collision_free_paths)}")
        print(f"📈 Collision-free candidate rate: {success_rate:.2f}%")

        if len(collision_free_paths) == 0:
            print("❌ Warning: No safe straight path found; all candidates may be obstructed by bone.")
            return None

        # Clinical selection criterion: choose the shortest insertion depth to minimize trauma.
        best_path = min(collision_free_paths, key=lambda p: p["path_length"])
        print(f">> 🏆 Best safe entry point: {[round(c, 2) for c in best_path['entry_point']]} mm")
        print(f">> 🏆 Target coordinates: {[round(c, 2) for c in target_pt]} mm")
        print(f">> 🏆 Best insertion depth: {best_path['path_length']:.2f} mm")
        print("==================================================\n")

        return best_path

if __name__ == "__main__":
    # Load the rib obstacles generated earlier.
    obs_file = r"data/demo_nodules/rib_obstacles_0037_segmented.json"
    
    # Physical nodule coordinates for patient 0037.
    target_0037 = [118.275, 20.625, -163.50]

    planner = BiopsyTrajectoryPlanner(obs_file)
    best_trajectory = planner.plan_multidirectional(target_0037)

    # Save the best plan for the next manipulator stage.
    if best_trajectory is not None:
        out_plan_file = r"F:\Lung-Nodule\data\demo_nodules\planned_trajectory_0037.json"
        with open(out_plan_file, "w", encoding="utf-8") as f:
            json.dump({
                "target_point": target_0037,
                "best_trajectory": best_trajectory
            }, f, indent=4)
        print(f"🎉 Planned path saved to: {out_plan_file}")
