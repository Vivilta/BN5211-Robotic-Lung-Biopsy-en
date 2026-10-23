from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient(
    host="localhost",
    port=23000
)

sim = client.require("sim")

robot = sim.getObject("/UR5")
needle = sim.getObject("/UR5/needle")

joints = sim.getObjectsInTree(
    robot,
    sim.object_joint_type,
    0
)

joint6 = joints[-1]

print("\n========== NEEDLE GEOMETRY ==========")

print("Needle position relative to Joint 6:")
print(sim.getObjectPosition(needle, joint6))

print("\nNeedle local bounding box:")

params = {
    "min_x": sim.objfloatparam_objbbox_min_x,
    "max_x": sim.objfloatparam_objbbox_max_x,
    "min_y": sim.objfloatparam_objbbox_min_y,
    "max_y": sim.objfloatparam_objbbox_max_y,
    "min_z": sim.objfloatparam_objbbox_min_z,
    "max_z": sim.objfloatparam_objbbox_max_z,
}

for name, param in params.items():
    value = sim.getObjectFloatParam(needle, param)
    print(f"{name}: {value:.6f} m")


print("\n========== JOINT 6 OBJECT TREE ==========")

objects = sim.getObjectsInTree(
    joint6,
    sim.handle_all,
    0
)

for obj in objects:
    alias = sim.getObjectAlias(obj)
    pos = sim.getObjectPosition(obj, joint6)

    print(f"{alias}: {pos}")


print("\n========== NEEDLE / WRIST COLLISION ==========")

for obj in objects:
    if obj == needle:
        continue

    alias = sim.getObjectAlias(obj)

    if "link6" in alias or "link7" in alias:
        result = sim.checkCollision(needle, obj)
        collided = (
            bool(result[0])
            if isinstance(result, (list, tuple))
            else bool(result)
        )

        print(f"Needle vs {alias}: {collided}")


print("\n========== NEEDLE VS ALL ROBOT LINKS ==========")

all_shapes = sim.getObjectsInTree(
    robot,
    sim.object_shape_type,
    0
)

for shape in all_shapes:
    if shape == needle:
        continue

    alias = sim.getObjectAlias(shape, 4)

    if "visible" not in alias:
        continue

    result = sim.checkCollision(needle, shape)

    collided = (
        bool(result[0])
        if isinstance(result, (list, tuple))
        else bool(result)
    )

    pos = sim.getObjectPosition(shape, joint6)

    print(
        f"{alias:25s} "
        f"Z={pos[2]*1000:8.2f} mm "
        f"Collision={collided}"
    )