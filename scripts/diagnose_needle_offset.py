from coppeliasim_zmqremoteapi_client import RemoteAPIClient

sim = RemoteAPIClient(
    host="localhost",
    port=23000
).require("sim")

robot = sim.getObject("/UR5")
needle = sim.getObject("/UR5/needle")

joints = sim.getObjectsInTree(
    robot, sim.object_joint_type, 0
)
joint6 = joints[-1]

shapes = sim.getObjectsInTree(
    robot, sim.object_shape_type, 0
)

link6 = next(
    s for s in shapes
    if sim.getObjectAlias(s, 4).startswith("link6_visible")
)

original_pose = sim.getObjectPose(needle, joint6)

try:
    print("Offset scan:")

    for center_z_mm in [
        -110, -120, -130, -140,
        -150, -170, -200,
        110, 120, 130, 150, 170, 200
    ]:
        pose = original_pose.copy()
        pose[2] = center_z_mm / 1000.0

        sim.setObjectPose(
            needle,
            pose,
            joint6
        )

        result = sim.checkCollision(
            needle, link6
        )

        collided = (
            bool(result[0])
            if isinstance(result, (list, tuple))
            else bool(result)
        )

        print(
            f"Center Z = {center_z_mm:>4} mm "
            f"Collision = {collided}"
        )

finally:
    sim.setObjectPose(
        needle,
        original_pose,
        joint6
    )