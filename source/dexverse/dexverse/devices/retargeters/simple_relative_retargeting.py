# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal hand-retargeter scaffold for custom relative teleoperation logic.

This module keeps the wrist logic intentionally simple while supporting
different robot action layouts, including bimanual robots that pack both hands
into a single action vector.

Three ``wrist_rot_repr`` modes are supported in the per-hand layout:

* ``"euler"`` – Relative Euler-angle displacement (3 floats). Suited for
  floating hands with virtual rotation joints.
* ``"rotvec"`` – Relative rotation-vector / axis-angle displacement (3 floats).
  Matches :func:`~isaaclab.utils.math.apply_delta_pose` as used by
  :class:`~isaaclab.controllers.differential_ik.DifferentialIKController` in
  ``pose`` + ``use_relative_mode=True``.
* ``"quat_absolute"`` – Absolute target quaternion ``(w, x, y, z)`` (4 floats).
  Used with ``DifferentialIKController(use_relative_mode=False)`` for arm-based
  robots. The layout must include ``ee_default_pose_b`` (7-tuple: pos + wxyz
  quat of the end-effector at the robot's home joint configuration, expressed
  in the robot base frame).  The target EE pose is computed as the home pose
  offset by the hand-tracking displacement from the calibration pose, giving
  position-control semantics (stationary hand → stationary robot).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import isaaclab.sim as sim_utils
import numpy as np
import torch
import yaml
from dex_retargeting.retargeting_config import RetargetingConfig
from dexverse.robot_agents import (
    SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES,
    SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES,
)
from dexverse.visual_purpose import hide_marker_from_cameras
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import retrieve_file_path
from scipy.spatial.transform import Rotation as R


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)


def _default_wrist_pose() -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    pose[3] = 1.0
    return pose


SIMPLE_RETARGETER_LAYOUT_SOURCES = SIMPLE_RELATIVE_ROBOT_LAYOUT_SOURCES

# Per-robot_type override for dex-retargeting dict attribute name on the layout module.
DEX_RETARGETING_ATTR_OVERRIDES = SIMPLE_RELATIVE_DEX_RETARGETING_ATTR_OVERRIDES

HAND_NAME_TO_TARGET = {
    "left": DeviceBase.TrackingTarget.HAND_LEFT,
    "right": DeviceBase.TrackingTarget.HAND_RIGHT,
}

DEX_RETARGETING_ATTR_NAME = "SIMPLE_RELATIVE_DEX_RETARGETING"
DEX_RETARGETING_HAND_JOINT_INDICES = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25)
DEX_RETARGETING_HAND_JOINT_NAMES = [HAND_JOINT_NAMES[idx] for idx in DEX_RETARGETING_HAND_JOINT_INDICES]

# Hard-coded per-hand rotation (degrees) around the wrist/canonical z-axis
# applied to the human finger keypoints before feeding them to dex-retargeting.
# Use this to compensate for the difference in "heading" between the human
# hand's forward axis and the robot hand's forward axis. Positive values rotate
# the finger reference points counter-clockwise when looking down the +z axis
# of the canonical wrist frame. Edit these values per hand as needed.
FINGER_Z_ROTATION_DEG: dict[str, float] = {
    "right": -15.0,
    "left": 15.0,
}


class SimpleRelativeRetargeter(RetargeterBase):
    """Minimal wrist retargeter with robot-type-driven action layout support."""

    def __init__(self, cfg: SimpleRelativeRetargeterCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._layout = self._get_action_layout(cfg.robot_type)
        self._tracked_hands = tuple(self._layout["hands"].keys())
        self._bound_hand = cfg.bound_hand

        expected_output_dim = int(self._layout["output_dim"])
        if cfg.output_dim < 0:
            raise ValueError(f"output_dim must be non-negative, got {cfg.output_dim}.")
        if cfg.output_dim == 0:
            self.cfg.output_dim = expected_output_dim
        elif cfg.output_dim < expected_output_dim:
            raise ValueError(
                f"output_dim={cfg.output_dim} is too small for robot_type='{cfg.robot_type}'. "
                f"Expected at least {expected_output_dim}."
            )

        self.latest_wrist_poses = {hand: None for hand in self._tracked_hands}
        self.retarget_base_wrist_poses = {hand: _default_wrist_pose() for hand in self._tracked_hands}
        self._dex_retgt = {}
        self._dex_output_joint_names = {}
        self._dex_to_action_finger_indices = {}
        self._dex_config_temp_paths: list[str] = []

        # Keep these aliases for the simple single-hand workflow already in use.
        self.lastest_wrist_pose: np.ndarray | None = None
        self.retarget_base_wrist_pose: np.ndarray = _default_wrist_pose()

        point_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/simple_relative_hand_keypoints",
            markers={
                "joint": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
            },
        )
        self._markers = VisualizationMarkers(point_marker_cfg)
        hide_marker_from_cameras(self._markers)
        self._markers.set_visibility(True)

        canonical_point_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/simple_relative_canonical_hand_keypoints",
            markers={
                "joint": sim_utils.SphereCfg(
                    radius=0.004,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 1.0)),
                )
            },
        )
        self._canonical_markers = VisualizationMarkers(canonical_point_marker_cfg)
        hide_marker_from_cameras(self._canonical_markers)
        self._canonical_markers.set_visibility(True)

        wrist_frame_cfg = FRAME_MARKER_CFG.copy()
        wrist_frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self._wrist_markers = VisualizationMarkers(
            wrist_frame_cfg.replace(prim_path="/Visuals/simple_relative_wrist_frames")
        )
        hide_marker_from_cameras(self._wrist_markers)
        self._wrist_markers.set_visibility(True)

        if cfg.initialize_dex_retargeting:
            self._initialize_dex_retargeters()

    def retarget(self, data: dict[DeviceBase.TrackingTarget, Any]) -> torch.Tensor:
        """Convert raw device data into a robot command tensor."""
        hand_data_by_target = {hand: self._get_hand_data(data, hand) for hand in self._tracked_hands}

        for hand, hand_data in hand_data_by_target.items():
            self.latest_wrist_poses[hand] = self._extract_wrist_pose(hand_data)

        self.lastest_wrist_pose = self.latest_wrist_poses.get(self._bound_hand)
        if self._bound_hand in self.retarget_base_wrist_poses:
            self.retarget_base_wrist_pose = self.retarget_base_wrist_poses[self._bound_hand].copy()

        self._visualize_hand_keypoints(hand_data_by_target)
        self._visualize_canonical_hand_keypoints(hand_data_by_target)
        self._visualize_wrist_poses()

        if self.cfg.default_command:
            action = np.array(self.cfg.default_command, dtype=np.float32).copy()
        else:
            action = np.zeros(self.cfg.output_dim, dtype=np.float32)

        for hand in self._tracked_hands:
            self._assign_hand_wrist_command(action, hand)
            finger_values = self._retarget_hand_fingers(hand_data_by_target[hand], hand)
            self._assign_hand_fingers(action, hand, finger_values)

        return torch.tensor(action, dtype=torch.float32, device=self._sim_device)

    def calibrate_wrist_pose(self):
        calibrated_hands = []
        missing_hands = []

        for hand in self._tracked_hands:
            wrist_pose = self.latest_wrist_poses.get(hand)
            if wrist_pose is None:
                missing_hands.append(hand.name)
                continue
            self.retarget_base_wrist_poses[hand] = wrist_pose.copy()
            calibrated_hands.append(hand.name)

        if not calibrated_hands:
            print("No wrist pose available yet; move the hand once before calibrating.")
            return

        if self._bound_hand in self.retarget_base_wrist_poses:
            self.retarget_base_wrist_pose = self.retarget_base_wrist_poses[self._bound_hand].copy()

        print(f"Calibrated wrist pose for {', '.join(calibrated_hands)}")
        if missing_hands:
            print(f"No wrist pose available for {', '.join(missing_hands)}")

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        """Request only the device features this retargeter needs."""
        return [RetargeterBase.Requirement.HAND_TRACKING]

    def _get_action_layout(self, robot_type: str) -> dict[str, Any]:
        if robot_type not in SIMPLE_RETARGETER_LAYOUT_SOURCES:
            raise ValueError(
                f"Unsupported robot_type '{robot_type}'. "
                f"Supported values: {sorted(SIMPLE_RETARGETER_LAYOUT_SOURCES.keys())}."
            )
        module_name, attr_name = SIMPLE_RETARGETER_LAYOUT_SOURCES[robot_type]
        module = import_module(module_name)
        raw_layout = getattr(module, attr_name)

        hands = {}
        for hand_name, hand_layout in raw_layout["hands"].items():
            if hand_name not in HAND_NAME_TO_TARGET:
                raise ValueError(f"Unsupported hand key '{hand_name}' in layout for robot_type='{robot_type}'.")
            hands[HAND_NAME_TO_TARGET[hand_name]] = hand_layout

        return {
            "output_dim": int(raw_layout["output_dim"]),
            "hands": hands,
        }

    def _get_robot_module(self):
        module_name, _ = SIMPLE_RETARGETER_LAYOUT_SOURCES[self.cfg.robot_type]
        return import_module(module_name)

    def _get_dex_retargeting_spec(self) -> dict[str, Any] | None:
        module = self._get_robot_module()
        attr_name = DEX_RETARGETING_ATTR_OVERRIDES.get(self.cfg.robot_type, DEX_RETARGETING_ATTR_NAME)
        return getattr(module, attr_name, None)

    def _initialize_dex_retargeters(self) -> None:
        """Load dex-retargeting configs and build retargeters for available hands."""
        dex_spec = self._get_dex_retargeting_spec()
        for hand_name, hand_spec in dex_spec.get("hands", {}).items():
            if hand_name not in HAND_NAME_TO_TARGET:
                raise ValueError(f"Unsupported dex-retargeting hand key '{hand_name}'.")

            config_path = self._resolve_dex_config_path(hand_name, hand_spec)
            urdf_path = hand_spec.get("urdf_path")
            if not config_path or not urdf_path:
                raise ValueError(
                    f"Dex-retargeting spec for hand '{hand_name}' must define a config path "
                    "(via 'config_paths' or legacy 'config_path') and 'urdf_path'."
                )

            local_urdf_path = retrieve_file_path(urdf_path, force_download=True)
            patched_config_path = self._create_dex_config_with_urdf(config_path, local_urdf_path)
            retargeter = RetargetingConfig.load_from_file(patched_config_path).build()

            hand_target = HAND_NAME_TO_TARGET[hand_name]
            self._dex_retgt[hand_target] = retargeter
            self._dex_output_joint_names[hand_target] = self._get_dex_output_joint_names(retargeter)
            self._dex_to_action_finger_indices[hand_target] = self._build_finger_name_mapping(hand_target)

    def _resolve_dex_config_path(self, hand_name: str, hand_spec: dict[str, Any]) -> str:
        """Pick the dex-retargeting YAML for the configured retargeting scheme.

        Two spec shapes are supported:

        * ``config_paths``: a dict keyed by scheme name (e.g. ``"dexpilot"`` /
          ``"vector"``) -> YAML path. Preferred; lets a single robot offer
          multiple retargeting schemes selectable via
          :attr:`SimpleRelativeRetargeterCfg.retargeting_scheme`.
        * ``config_path``: a single YAML path (legacy). Always treated as the
          ``"dexpilot"`` scheme.
        """
        scheme = getattr(self.cfg, "retargeting_scheme", "dexpilot")
        config_paths = hand_spec.get("config_paths")
        if config_paths:
            if scheme not in config_paths:
                raise ValueError(
                    f"No dex-retargeting config for scheme '{scheme}' "
                    f"(hand '{hand_name}', robot_type='{self.cfg.robot_type}'). "
                    f"Available schemes: {sorted(config_paths)}."
                )
            return config_paths[scheme]

        legacy_path = hand_spec.get("config_path")
        if legacy_path and scheme == "dexpilot":
            return legacy_path
        raise ValueError(
            f"Dex-retargeting spec for hand '{hand_name}' (robot_type='{self.cfg.robot_type}') "
            f"has no config for scheme '{scheme}'. Add a 'config_paths' dict that includes this "
            "scheme, or use scheme 'dexpilot' with a legacy 'config_path'."
        )

    def _create_dex_config_with_urdf(self, config_path: str, urdf_path: str) -> str:
        """Create a temporary dex-retargeting config with the resolved URDF path."""
        with open(config_path) as file:
            config = yaml.safe_load(file)

        if "retargeting" not in config:
            raise ValueError(f"Invalid dex-retargeting config '{config_path}': missing 'retargeting' section.")

        config["retargeting"]["urdf_path"] = urdf_path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp_file:
            yaml.safe_dump(config, temp_file)
            temp_path = temp_file.name

        self._dex_config_temp_paths.append(temp_path)
        return temp_path

    def _get_dex_output_joint_names(self, retargeter) -> list[str]:
        """Return dex-retargeting output joint names in the same order as retarget().

        ``SeqRetargeting.retarget()`` scatters the optimised target-joint values
        back into a full ``robot_qpos`` array indexed by pinocchio DOF order
        (``robot.dof_joint_names``), so the output is always in URDF/DOF order —
        **not** ``optimizer.target_joint_names`` (YAML) order.
        """
        return [str(name) for name in retargeter.optimizer.robot.dof_joint_names]

    def _get_hand_data(self, data: dict[DeviceBase.TrackingTarget, Any], hand: DeviceBase.TrackingTarget) -> Any:
        """Fetch one hand's tracking payload from Isaac Lab data."""
        return data.get(hand, {})

    def _extract_wrist_pose(self, hand_data: Any) -> np.ndarray:
        """Extract the wrist pose as a 7D numpy array if present."""
        if not isinstance(hand_data, dict):
            return _default_wrist_pose()

        wrist_pose = hand_data.get("wrist")
        if wrist_pose is None:
            return _default_wrist_pose()

        return np.asarray(wrist_pose, dtype=np.float32).copy()

    def _layout_rotation(self, hand_layout: dict[str, Any], key: str) -> R:
        quat_wxyz = hand_layout.get(key)
        if quat_wxyz is None:
            return R.identity()
        return R.from_quat(wxyz_to_xyzw(np.asarray(quat_wxyz, dtype=np.float32)))

    def _express_relative_wrist_rotation_in_action_frame(
        self,
        relative_rot: R,
        hand_layout: dict[str, Any],
    ) -> R:
        wrist_base_rot = self._layout_rotation(hand_layout, "wrist_base_rot")
        return wrist_base_rot.inv() * relative_rot * wrist_base_rot

    def _reorder_wrist_euler(self, euler_xyz: np.ndarray, rotation_order: str) -> np.ndarray:
        if rotation_order == "xyz":
            return np.array(euler_xyz, dtype=np.float32)
        if rotation_order == "yaw_pitch_roll":
            return np.array([euler_xyz[2], euler_xyz[1], euler_xyz[0]], dtype=np.float32)
        raise ValueError(f"Unsupported wrist rotation order '{rotation_order}'.")

    def _compute_relative_wrist_euler(
        self,
        wrist_quat_wxyz: np.ndarray,
        base_quat_wxyz: np.ndarray,
        hand_layout: dict[str, Any],
    ) -> np.ndarray:
        """Compute relative wrist rotation in the target robot action order."""
        wrist_rot = self._get_normalized_wrist_rotation(wrist_quat_wxyz)
        base_rot = self._get_normalized_wrist_rotation(base_quat_wxyz)
        relative_rot = wrist_rot * base_rot.inv()
        relative_rot = self._express_relative_wrist_rotation_in_action_frame(relative_rot, hand_layout)

        euler_xyz = relative_rot.as_euler("XYZ", degrees=False)
        return self._reorder_wrist_euler(euler_xyz, hand_layout["wrist_rot_order"])

    def _compute_relative_wrist_rotvec(
        self,
        wrist_quat_wxyz: np.ndarray,
        base_quat_wxyz: np.ndarray,
        hand_layout: dict[str, Any],
    ) -> np.ndarray:
        """Relative wrist rotation as rotation vector (axis-angle), radians.

        Matches ``DifferentialIKController`` relative pose commands: the last three
        action components are interpreted as orientation displacement in axis-angle
        form by :func:`isaaclab.utils.math.apply_delta_pose`.
        """
        wrist_rot = self._get_normalized_wrist_rotation(wrist_quat_wxyz)
        base_rot = self._get_normalized_wrist_rotation(base_quat_wxyz)
        relative_rot = wrist_rot * base_rot.inv()
        relative_rot = self._express_relative_wrist_rotation_in_action_frame(relative_rot, hand_layout)
        return np.asarray(relative_rot.as_rotvec(), dtype=np.float32)

    def _get_normalized_wrist_rotation(self, wrist_quat_wxyz: np.ndarray) -> R:
        """Normalize OpenXR wrist orientation to a stable hand-centric frame."""
        wrist_rotation_raw = R.from_quat(wxyz_to_xyzw(wrist_quat_wxyz))
        x_plus_90 = R.from_euler("x", -90, degrees=True)
        y_minus_90 = R.from_euler("y", 90, degrees=True)
        normalize_rotation = y_minus_90 * x_plus_90
        return wrist_rotation_raw * normalize_rotation

    def _assign(self, action: np.ndarray, indices: tuple[int, ...], values: np.ndarray) -> None:
        """Assign values into the action vector using an explicit index mapping."""
        idx_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        val_arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if idx_arr.size == 0 or val_arr.size == 0:
            return

        n = min(idx_arr.size, val_arr.size)
        idx_arr = idx_arr[:n]
        val_arr = val_arr[:n]
        valid_mask = (idx_arr >= 0) & (idx_arr < action.shape[0])
        if np.any(valid_mask):
            action[idx_arr[valid_mask]] = val_arr[valid_mask]

    def _assign_hand_wrist_command(self, action: np.ndarray, hand: DeviceBase.TrackingTarget) -> None:
        """Write one hand's wrist translation/rotation into the action vector."""
        wrist_pose = self.latest_wrist_poses.get(hand)
        base_pose = self.retarget_base_wrist_poses.get(hand)
        if wrist_pose is None or base_pose is None:
            return

        wrist_pos = np.asarray(wrist_pose[:3], dtype=np.float32).copy()
        wrist_quat = np.asarray(wrist_pose[3:], dtype=np.float32).copy()

        hand_layout = self._layout["hands"][hand]
        rot_repr = hand_layout.get("wrist_rot_repr", "euler")

        if rot_repr == "quat_absolute":
            self._assign_absolute_wrist_command(
                action,
                hand_layout,
                wrist_pos,
                wrist_quat,
                base_pose,
            )
            return

        wrist_abs_pos_cmd = wrist_pos - base_pose[:3]
        wrist_base_rot = self._layout_rotation(hand_layout, "wrist_base_rot")
        wrist_abs_pos_cmd = wrist_base_rot.inv().apply(wrist_abs_pos_cmd).astype(np.float32)

        if rot_repr == "euler":
            relative_rot_cmd = self._compute_relative_wrist_euler(
                wrist_quat,
                base_pose[3:],
                hand_layout,
            )
        elif rot_repr == "rotvec":
            relative_rot_cmd = self._compute_relative_wrist_rotvec(wrist_quat, base_pose[3:], hand_layout)
        else:
            raise ValueError(f"Unsupported wrist_rot_repr '{rot_repr}'. Use 'euler', 'rotvec', or 'quat_absolute'.")
        rot_signs = np.asarray(hand_layout.get("wrist_rot_signs", (1.0, 1.0, 1.0)), dtype=np.float32)
        relative_rot_cmd = relative_rot_cmd * rot_signs
        self._assign(action, hand_layout["wrist_trans_indices"], wrist_abs_pos_cmd)
        self._assign(action, hand_layout["wrist_rot_indices"], relative_rot_cmd)

    def _assign_absolute_wrist_command(
        self,
        action: np.ndarray,
        hand_layout: dict[str, Any],
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
        base_pose: np.ndarray,
    ) -> None:
        """Write an absolute EE target pose for arm-based IK controllers.

        Used with ``DifferentialIKController(use_relative_mode=False)``.
        The target is computed as the robot's home EE pose offset by the
        hand-tracking displacement from the calibration pose, giving
        position-control semantics (stationary hand → stationary robot).
        """
        ee_default = np.asarray(hand_layout["ee_default_pose_b"], dtype=np.float32)
        ee_default_pos = ee_default[:3]
        ee_default_quat_wxyz = ee_default[3:]

        target_pos = ee_default_pos + (wrist_pos - base_pose[:3])

        wrist_rot = self._get_normalized_wrist_rotation(wrist_quat_wxyz)
        base_rot = self._get_normalized_wrist_rotation(base_pose[3:])
        relative_rot = wrist_rot * base_rot.inv()

        ee_default_rot = R.from_quat(wxyz_to_xyzw(ee_default_quat_wxyz))
        target_rot = relative_rot * ee_default_rot
        target_quat_wxyz = xyzw_to_wxyz(target_rot.as_quat().astype(np.float32))

        self._assign(action, hand_layout["wrist_trans_indices"], target_pos)
        self._assign(action, hand_layout["wrist_rot_indices"], target_quat_wxyz)

    def _assign_hand_fingers(
        self,
        action: np.ndarray,
        hand: DeviceBase.TrackingTarget,
        finger_values: np.ndarray,
    ) -> None:
        """Assign finger values through the configured per-hand mapping.

        This is not used yet, but keeps the mapping logic in one simple place for
        when finger retargeting is added later.
        """
        hand_layout = self._layout["hands"][hand]
        finger_indices = tuple(hand_layout["finger_indices"])
        finger_values = np.asarray(finger_values, dtype=np.float32).reshape(-1)
        if not finger_indices or not finger_values.size:
            return

        name_mapping = self._dex_to_action_finger_indices.get(hand)
        if name_mapping is not None and len(name_mapping) == len(finger_indices):
            ordered_finger_values = np.zeros(len(name_mapping), dtype=np.float32)
            for action_idx, dex_idx in enumerate(name_mapping):
                if dex_idx is not None and dex_idx < finger_values.size:
                    ordered_finger_values[action_idx] = finger_values[dex_idx]
            self._assign(action, finger_indices, ordered_finger_values)
            return

        finger_permutation = np.asarray(hand_layout["finger_permutation"], dtype=np.int64)
        if finger_permutation.size:
            self._assign(action, finger_indices, finger_values[finger_permutation])

    def _retarget_hand_fingers(self, hand_data: Any, hand: DeviceBase.TrackingTarget) -> np.ndarray:
        """Run minimal DexPilot finger retargeting for one hand."""
        retargeter = self._dex_retgt.get(hand)
        if retargeter is None or not isinstance(hand_data, dict):
            return np.zeros(0, dtype=np.float32)

        canonical_joint_positions = self._convert_hand_to_canonical_joint_positions(hand_data, hand)
        if canonical_joint_positions is None:
            return np.zeros(len(self._dex_output_joint_names.get(hand, [])), dtype=np.float32)

        ref_value = self._compute_dex_ref_value(retargeter, canonical_joint_positions)
        retarget_kwargs: dict[str, Any] = {}
        optimizer = getattr(retargeter, "optimizer", None)
        fixed_indices = getattr(optimizer, "idx_pin2fixed", None)
        if fixed_indices is not None:
            fixed_indices = np.asarray(fixed_indices, dtype=np.int64).reshape(-1)
            if fixed_indices.size:
                # Keep non-target joints (e.g. floating wrist joints in Shadow retarget URDFs)
                # at their neutral pose so newer dex-retargeting versions receive the expected
                # fixed_qpos / non_target_qpos payload.
                retarget_kwargs["fixed_qpos"] = np.zeros(fixed_indices.size, dtype=np.float32)
        with torch.enable_grad():
            with torch.inference_mode(False):
                finger_values = retargeter.retarget(ref_value, **retarget_kwargs)
        return np.asarray(finger_values, dtype=np.float32)

    def _build_finger_name_mapping(self, hand: DeviceBase.TrackingTarget) -> list[int | None]:
        """Map dex-retargeting output joints into the robot action finger order by name."""
        hand_layout = self._layout["hands"][hand]
        desired_joint_names = tuple(hand_layout.get("finger_joint_names", ()))
        if not desired_joint_names:
            return []

        dex_output_joint_names = self._dex_output_joint_names.get(hand, [])
        mapping = []
        for desired_name in desired_joint_names:
            try:
                mapping.append(dex_output_joint_names.index(desired_name))
            except ValueError:
                mapping.append(None)
        return mapping

    def _convert_hand_to_canonical_joint_positions(
        self,
        hand_data: dict[str, Any],
        hand: DeviceBase.TrackingTarget | None = None,
    ) -> np.ndarray | None:
        """Convert raw OpenXR hand joints into wrist-centered canonical coordinates.

        If a non-zero entry is configured in :data:`FINGER_Z_ROTATION_DEG` for
        this hand, an additional rotation around the canonical wrist z-axis is
        applied to the joint positions to compensate for the human/robot hand
        heading difference.
        """
        wrist_pose = hand_data.get("wrist")
        if wrist_pose is None:
            return None

        wrist_pose = np.asarray(wrist_pose, dtype=np.float32)
        wrist_pos = wrist_pose[:3]
        wrist_quat = wrist_pose[3:]
        canonical_rotation = self._get_normalized_wrist_rotation(wrist_quat).as_matrix().astype(np.float32)

        joint_positions = np.zeros((len(DEX_RETARGETING_HAND_JOINT_NAMES), 3), dtype=np.float32)
        for idx, joint_name in enumerate(DEX_RETARGETING_HAND_JOINT_NAMES):
            joint_pose = hand_data.get(joint_name)
            if joint_pose is None:
                return None
            joint_positions[idx] = np.asarray(joint_pose, dtype=np.float32)[:3]

        joint_positions = joint_positions - wrist_pos[None, :]
        canonical_positions = joint_positions @ canonical_rotation

        hand_key = hand.name.replace("HAND_", "").lower() if hand is not None else None
        z_rot_deg = FINGER_Z_ROTATION_DEG.get(hand_key, 0.0) if hand_key is not None else 0.0
        if z_rot_deg != 0.0:
            z_rot_matrix = R.from_euler("z", z_rot_deg, degrees=True).as_matrix().astype(np.float32)
            # Row-vector convention: `p_row @ M` rotates points by M^T in the
            # canonical frame, equivalent to re-expressing them in a frame
            # rotated by +z_rot_deg around the wrist's z-axis.
            canonical_positions = canonical_positions @ z_rot_matrix

        return canonical_positions

    def _compute_dex_ref_value(self, retargeter, canonical_joint_positions: np.ndarray) -> np.ndarray:
        """Build the DexPilot reference value from canonical human joint positions."""
        optimizer = retargeter.optimizer
        indices = optimizer.target_link_human_indices
        if optimizer.retargeting_type == "POSITION":
            ref_value = canonical_joint_positions[indices, :]
            # PositionOptimizer stores its target link names on ``body_names``.
            target_link_names = getattr(optimizer, "body_names", None)
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = canonical_joint_positions[task_indices, :] - canonical_joint_positions[origin_indices, :]
            # Vector / DexPilot optimizers store theirs on ``task_link_names``.
            target_link_names = getattr(optimizer, "task_link_names", None)

        ref_value = self._apply_finger_scales(ref_value, target_link_names)
        return ref_value

    def _apply_finger_scales(
        self,
        ref_value: np.ndarray,
        target_link_names: list[str] | None,
    ) -> np.ndarray:
        """Multiply ref-value rows by per-finger scales from ``cfg.finger_scales``.

        Each row's target link name is matched against ``cfg.finger_scales`` by
        longest-prefix; rows with no matching prefix are left at scale 1.0.
        Returns the original array when the scales dict is empty so the
        common path stays allocation-free.
        """
        finger_scales = getattr(self.cfg, "finger_scales", None) or {}
        if not finger_scales or not target_link_names:
            return ref_value

        num_rows = ref_value.shape[0]
        if len(target_link_names) != num_rows:
            # Mismatch (older optimizer / unexpected layout) — skip scaling
            # rather than risk mis-applying values to the wrong finger.
            return ref_value

        row_scales = np.ones(num_rows, dtype=ref_value.dtype)
        for i, link_name in enumerate(target_link_names):
            scale = self._lookup_finger_scale(link_name, finger_scales)
            if scale is not None:
                row_scales[i] = float(scale)
        if np.all(row_scales == 1.0):
            return ref_value
        return ref_value * row_scales[:, None]

    @staticmethod
    def _lookup_finger_scale(link_name: str, finger_scales: dict[str, float]) -> float | None:
        """Return the scale for the longest prefix of ``link_name`` in ``finger_scales``."""
        if not isinstance(link_name, str):
            return None
        best_prefix: str | None = None
        for prefix in finger_scales:
            if not isinstance(prefix, str) or not link_name.startswith(prefix):
                continue
            if best_prefix is None or len(prefix) > len(best_prefix):
                best_prefix = prefix
        return finger_scales[best_prefix] if best_prefix is not None else None

    def _visualize_hand_keypoints(self, hand_data_by_target: dict[DeviceBase.TrackingTarget, Any]) -> None:
        """Visualize the current hand keypoints as small spheres."""
        joint_positions = []
        for hand_data in hand_data_by_target.values():
            if not isinstance(hand_data, dict):
                continue
            for pose in hand_data.values():
                pose_array = np.asarray(pose, dtype=np.float32)
                if pose_array.shape[0] >= 3:
                    joint_positions.append(pose_array[:3])

        if not joint_positions:
            return

        self._markers.visualize(translations=np.asarray(joint_positions, dtype=np.float32))

    def _visualize_canonical_hand_keypoints(
        self,
        hand_data_by_target: dict[DeviceBase.TrackingTarget, Any],
    ) -> None:
        """Visualize canonicalized hand joints, translated back to the wrist position."""
        canonical_joint_positions_world = []
        for hand, hand_data in hand_data_by_target.items():
            if not isinstance(hand_data, dict):
                continue

            canonical_joint_positions = self._convert_hand_to_canonical_joint_positions(hand_data, hand)
            wrist_pose = hand_data.get("wrist")
            if canonical_joint_positions is None or wrist_pose is None:
                continue

            wrist_pos = np.asarray(wrist_pose, dtype=np.float32)[:3]
            canonical_joint_positions_world.append(canonical_joint_positions + wrist_pos[None, :])

        if not canonical_joint_positions_world:
            return

        self._canonical_markers.visualize(
            translations=np.concatenate(canonical_joint_positions_world, axis=0).astype(np.float32)
        )

    def _visualize_wrist_poses(self) -> None:
        """Visualize tracked wrist poses as frame markers."""
        wrist_positions = []
        wrist_orientations = []
        for hand in self._tracked_hands:
            wrist_pose = self.latest_wrist_poses.get(hand)
            if wrist_pose is None:
                continue
            wrist_pose = np.asarray(wrist_pose, dtype=np.float32)
            wrist_positions.append(wrist_pose[:3])
            wrist_orientations.append(wrist_pose[3:])

        if not wrist_positions:
            return

        self._wrist_markers.visualize(
            translations=np.asarray(wrist_positions, dtype=np.float32),
            orientations=np.asarray(wrist_orientations, dtype=np.float32),
        )


@dataclass
class SimpleRelativeRetargeterCfg(RetargeterCfg):
    """Configuration for `SimpleRelativeRetargeter`.

    Attributes:
        robot_type: Action layout to use. This determines which hands are read
            and where their commands are written in the action vector.
        bound_hand: Main hand for the simple single-hand workflow and alias
            fields such as `lastest_wrist_pose`.
        output_dim: Size of the command vector returned by `retarget()`. If 0,
            the default size for `robot_type` is used.
        default_command: Optional placeholder action. If left empty, a zero
            vector of size `output_dim` is returned.
        finger_scales: Per-finger override for the ref-value scaling applied
            before the dex-retargeting optimizer runs. Keys are matched as
            longest-prefix against each row's target link name (e.g.
            ``optimizer.task_link_names``: ``"thtip"``, ``"thmiddle"`` for
            Shadow; ``"thumb_tip_head"`` for Leap). Values multiply the
            corresponding rows of the ref-value (palm→tip vectors for vector
            / DexPilot optimizers; per-link target positions for POSITION
            optimizers). Useful when the robot's finger lengths don't all
            match the human's at a single global scale — for example, the
            Shadow thumb is shorter than the human thumb relative to the
            wrist, so a value like ``{"th": 0.9, "ff": 1.1, "mf": 1.1,
            "rf": 1.1, "lf": 1.1}`` lets the index/middle/ring/little fingers
            run at 1.1 while keeping the thumb compressed to 0.9 so it still
            bends. Layered on top of the YAML ``scaling_factor`` (multiplies
            after this dict): leave that at 1.0 if you only want per-finger
            control. Empty dict (the default) preserves prior behavior.
        retargeting_scheme: Which dex-retargeting YAML to load when the robot
            module's dex spec exposes a ``config_paths`` dict keyed by scheme
            (e.g. ``"dexpilot"`` / ``"vector"``). Defaults to ``"dexpilot"``.
            Ignored for specs that only define a single legacy ``config_path``
            (those always behave as ``"dexpilot"``).
    """

    robot_type: str = "floating_shadow_right"
    bound_hand: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.HAND_RIGHT
    initialize_dex_retargeting: bool = True
    output_dim: int = 0
    default_command: tuple[float, ...] = field(default_factory=tuple)
    finger_scales: dict[str, float] = field(default_factory=dict)
    retargeting_scheme: str = "dexpilot"
    retargeter_type: type[RetargeterBase] = SimpleRelativeRetargeter
