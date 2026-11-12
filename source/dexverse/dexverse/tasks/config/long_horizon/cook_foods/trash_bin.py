"""Small open-top bin with a physical base and four separate wall colliders."""
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone

from .disposal import BIN_HALF_XY, BIN_HEIGHT, BIN_POSITION, BIN_WALL


@clone
def spawn_trash_bin(prim_path, cfg, translation=None, orientation=None):
    from isaacsim.core.utils.prims import create_prim

    root = create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    x, y = BIN_HALF_XY
    t, h = BIN_WALL, BIN_HEIGHT
    wall_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.22, 0.28), roughness=0.7)
    pieces = {
        "Base": ((2 * (x + t), 2 * (y + t), t), (0, 0, t / 2)),
        "Front": ((t, 2 * (y + t), h - t), (-x - t / 2, 0, (h + t) / 2)),
        "Back": ((t, 2 * (y + t), h - t), (x + t / 2, 0, (h + t) / 2)),
        "Left": ((2 * x, t, h - t), (0, y + t / 2, (h + t) / 2)),
        "Right": ((2 * x, t, h - t), (0, -y - t / 2, (h + t) / 2)),
    }
    for name, (size, pos) in pieces.items():
        box = sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                                 visual_material=wall_material,
                                 physics_material=sim_utils.RigidBodyMaterialCfg(
                                     static_friction=0.8, dynamic_friction=0.6, restitution=0.0))
        box.func(f"{prim_path}/{name}", box, translation=pos)
    # A light rim makes the open mouth easy to aim at, without a lid/collider.
    rim = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.8, 0.82))
    for name, (size, pos) in pieces.items():
        if name == "Base":
            continue
        trim = sim_utils.CuboidCfg(size=(size[0], size[1], 0.012), visual_material=rim)
        trim.func(f"{prim_path}/{name}Rim", trim, translation=(pos[0], pos[1], h - 0.006))
    sim_utils.define_rigid_body_properties(prim_path, sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True, disable_gravity=True))
    sim_utils.define_mass_properties(prim_path, sim_utils.MassPropertiesCfg(mass=2.0))
    return root


def trash_bin_cfg():
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/TrashBin", spawn=SpawnerCfg(func=spawn_trash_bin),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=BIN_POSITION))
