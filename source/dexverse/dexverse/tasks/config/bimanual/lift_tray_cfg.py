# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual lift of a flat tray (synthesis/tray001).

The tray carries a decorative ``micro_food`` prop (beef stew) so the scene
reads as a real tray of food.  The food is a *separate* rigid body resting on
the tray rather than a fixed decoration: it can slide or spill if the tray is
tilted, which reinforces the "keep the tray level" success criterion.
"""

from dexverse.assets import SYNTHESIS_DIR
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .base_cfg import (
    BimanualLiftObjectEnvFloatingShadowBimanualCfg,
    make_lift_object_cfg,
)

TRAY_USD_PATH = str(SYNTHESIS_DIR / "tray001" / "model_redtray.usd")
TRAY_SUCCESS_MAX_TILT_RAD = 0.174532925  # 10 degrees.

# --- micro_food prop placed on the tray ----------------------------------
# Use the physics-prepped variant: it references the heavy mesh and already
# carries a convex-hull collider, matching what ``make_lift_object_cfg``
# expects (it re-applies the top-level rigid body and assumes collision is
# already authored).
FOOD_USD_PATH = str(
    SYNTHESIS_DIR / "micro_food" / "Meshy_AI_Beef_stew_with_vegeta_0520120030_texture__convexHull_rigidprep.usd"
)
FOOD_SCALE: tuple[float, float, float] = (2.0, 2.0, 2.0)
FOOD_MASS: float = 0.15
# Lay the prop flat: its authored thin axis (y, ~2cm) becomes vertical.
# Quaternion is 90 deg about x, ordered (w, x, y, z).
FOOD_REST_QUAT: tuple[float, float, float, float] = (
    0.7071067811865476,
    0.7071067811865476,
    0.0,
    0.0,
)
# Height of the prop centre above the tray root when it is (re)seated; it
# then settles the last centimetre onto the tray surface under gravity.
FOOD_Z_OFFSET: float = 0.03


@configclass
class LiftTrayEnvFloatingShadowBimanualCfg(BimanualLiftObjectEnvFloatingShadowBimanualCfg):
    """Lift a flat tray off the tabletop with two Shadow hands."""

    usd_path: str = TRAY_USD_PATH
    # Tune to the authored mesh extent.  The tray meshes are typically small
    # relative to the tabletop; scale up until the tray spans a useful area.
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mass: float = 0.4
    # Tray is flat; the root tends to sit on the geometric bottom.
    object_half_height: float = 0.01
    table_clearance: float = 0.02
    object_init_x_offset: float = 0.05
    object_init_y_offset: float = 0.0
    # positive 90 degrees around z-axis (yaw), w,x,y,z -- keeps the tray flat.
    object_init_rot: tuple[float, float, float, float] = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
    lift_height: float = 0.2
    # Trays are symmetric enough to allow modest yaw randomisation.
    obj_x_range: tuple[float, float] = (-0.03, 0.03)
    obj_y_range: tuple[float, float] = (-0.1, 0.1)
    obj_z_range: tuple[float, float] = (0.0, 0.0)
    obj_roll_range: tuple[float, float] = (0.0, 0.0)
    obj_pitch_range: tuple[float, float] = (0.0, 0.0)
    obj_yaw_range: tuple[float, float] = (-0.2, 0.2)

    # Success requires both lifting the tray and keeping its top surface nearly level.
    success_max_tilt_rad: float = TRAY_SUCCESS_MAX_TILT_RAD

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.func = mdp.object_upright_and_lifted
        self.terminations.success.params = {
            "object_cfg": SceneEntityCfg("object"),
            "min_height": self.lift_height,
            "max_tilt_rad": self.success_max_tilt_rad,
        }

        # --- Add the micro_food prop resting on the tray ------------------
        # A separate rigid body (prim ".../Food") so it behaves like real
        # cargo.  ``make_lift_object_cfg`` hardcodes the ".../Object" prim
        # path, so override it to avoid clobbering the tray.
        food_cfg = make_lift_object_cfg(
            usd_path=FOOD_USD_PATH,
            scale=FOOD_SCALE,
            mass=FOOD_MASS,
            init_rot=FOOD_REST_QUAT,
        )
        food_cfg.prim_path = "{ENV_REGEX_NS}/Food"
        # Initial (pre-first-reset) spawn: sit it just above the tray spawn
        # pose that the base class already computed for the object.
        ox, oy, oz = self.scene.object.init_state.pos
        food_cfg.init_state.pos = (ox, oy, oz + FOOD_Z_OFFSET)
        self.scene.food = food_cfg

        # Re-seat the food on the (randomised) tray at every reset.  Zero its
        # velocity first, then snap its pose onto the tray.  These run after
        # `reset_object`/`reset_success_marker`, so the tray pose is final.
        self.events.reset_food_velocity = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("food"),
            },
        )
        self.events.reset_food = EventTerm(
            func=mdp.sync_object,
            mode="reset",
            params={
                "target_cfg": SceneEntityCfg("food"),
                "source_cfg": SceneEntityCfg("object"),
                "z_offset": FOOD_Z_OFFSET,
                "quat": FOOD_REST_QUAT,
            },
        )
