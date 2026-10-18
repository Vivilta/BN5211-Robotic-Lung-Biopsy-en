"""Small, explicit geometry helpers shared by the analytical and CoppeliaSim paths.

All medical-image and analytical-robot coordinates in this project use millimetres.
CoppeliaSim uses metres, so conversion happens at the remote-API boundary only.
"""

import numpy as np


def as_homogeneous_transform(matrix):
    """Validate and return a 4x4 homogeneous transform as float64."""
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("The last row of a homogeneous transform must be [0, 0, 0, 1]")
    return transform


def transform_point(transform, point):
    """Transform a 3D point with a 4x4 matrix."""
    transform = as_homogeneous_transform(transform)
    point = np.asarray(point, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"Expected a 3D point, got {point.shape}")
    return (transform[:3, :3] @ point) + transform[:3, 3]


def transform_direction(transform, direction):
    """Transform a direction without applying the translation component."""
    transform = as_homogeneous_transform(transform)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,):
        raise ValueError(f"Expected a 3D direction, got {direction.shape}")
    result = transform[:3, :3] @ direction
    norm = np.linalg.norm(result)
    if norm == 0:
        raise ValueError("Direction must be non-zero")
    return result / norm


def identity_transform():
    """Return an explicit identity patient-to-robot transform."""
    return np.eye(4, dtype=float)


def overhead_mount_base_to_world(mount_height_mm=0.0):
    """UR5 base frame for a ceiling mount, expressed in scene millimetres."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[2, 3] = float(mount_height_mm)
    return transform


def patient_to_robot_base_for_mount_height(patient_to_robot_base, mount_height_mm):
    """Re-express a demo patient transform for a vertically shifted ceiling mount."""
    transform = as_homogeneous_transform(patient_to_robot_base)
    old_base_to_world = overhead_mount_base_to_world()
    new_base_to_world = overhead_mount_base_to_world(mount_height_mm)
    return np.linalg.inv(new_base_to_world) @ old_base_to_world @ transform


def rotation_from_z_axis(z_axis):
    """Build a right-handed rotation whose local Z axis is ``z_axis``."""
    z_axis = np.asarray(z_axis, dtype=float)
    norm = np.linalg.norm(z_axis)
    if norm == 0:
        raise ValueError("The tool axis must be non-zero")
    z_axis = z_axis / norm

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def rotation_with_z_axis_roll(z_axis, roll_rad):
    """Rotate a tool frame around its local Z axis without changing needle direction."""
    roll = float(roll_rad)
    cosine = np.cos(roll)
    sine = np.sin(roll)
    local_roll = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return rotation_from_z_axis(z_axis) @ local_roll


def rotation_to_quaternion_xyzw(rotation):
    """Convert a proper 3x3 rotation matrix to CoppeliaSim xyzw quaternion order."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation, got {rotation.shape}")

    trace = np.trace(rotation)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s

    quaternion = np.array([x, y, z, w], dtype=float)
    return quaternion / np.linalg.norm(quaternion)


def quaternion_xyzw_to_rotation(quaternion):
    """Convert a CoppeliaSim xyzw quaternion to a 3x3 rotation matrix."""
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected a four-value quaternion, got {quaternion.shape}")
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Quaternion must be non-zero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )

