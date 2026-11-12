"""NutThread accepts either axial orientation at the same physical depth."""

import __future__
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "source/dexverse/dexverse/tasks"


def _rotate(quat, vector, inverse=False):
    # Independent rotation implementation for the simulator-free fixture.
    rotations = Rotation.from_quat(quat.numpy()[:, [1, 2, 3, 0]])
    result = rotations.apply(vector.numpy(), inverse=inverse)
    return torch.as_tensor(result, dtype=vector.dtype)


@pytest.fixture
def success_check():
    path = TASKS / "mdp/terminations.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "nutthread_success")
    scope = {
        "torch": torch, "quat_apply": _rotate,
        "quat_apply_inverse": lambda q, v: _rotate(q, v, inverse=True),
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    cfg_path = TASKS / "config/contact_rich/nutthread_cfg.py"
    constants = {}
    names = {
        "ASSET_SCALE", "NUT_ASSET_SCALE", "BOLT_HEAD_HEIGHT", "BOLT_SHANK_HEIGHT",
        "THREAD_PITCH", "NUT_BASE_HEIGHT", "NUT_TOP_HEIGHT", "TARGET_THREAD_TURNS",
        "SUCCESS_THRESHOLD_TURNS", "CENTER_DIST_THRESH", "NUT_AXIS_OFFSET",
    }
    for assignment in ast.parse(cfg_path.read_text()).body:
        if isinstance(assignment, ast.Assign) and isinstance(assignment.targets[0], ast.Name):
            if assignment.targets[0].id in names:
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(cfg_path), "exec"), constants)
    params = {
        "nut_base_height": constants["NUT_BASE_HEIGHT"],
        "nut_top_height": constants["NUT_TOP_HEIGHT"],
        "nut_axis_offset": constants["NUT_AXIS_OFFSET"],
        "bolt_head_height": constants["BOLT_HEAD_HEIGHT"],
        "bolt_shank_height": constants["BOLT_SHANK_HEIGHT"],
        "thread_pitch": constants["THREAD_PITCH"],
        "target_thread_turns": constants["TARGET_THREAD_TURNS"],
        "success_threshold_turns": constants["SUCCESS_THRESHOLD_TURNS"],
        "center_dist_thresh": constants["CENTER_DIST_THRESH"],
    }
    return scope["nutthread_success"], params


def _env(params, *, flipped, bolt_angles, tilt=0):
    depths = np.array([0.006, 0.0155, 0.016, 0.030, 0.016])
    count = len(depths)
    bolt_rotation = Rotation.from_euler("xyz", bolt_angles, degrees=True)
    nut_rotation = bolt_rotation * Rotation.from_euler("x", (180 if flipped else 0) + tilt, degrees=True)
    bolt_pos = np.tile([1.5, -2.0, 0.6], (count, 1))
    entry_bolt = np.zeros((count, 3))
    entry_bolt[:, 2] = params["bolt_head_height"] + params["bolt_shank_height"] - depths
    entry_bolt[-1, 0] = 0.003  # Deep, but not centered.
    face_height = params["nut_top_height"] if flipped else params["nut_base_height"]
    nut_pos = bolt_pos + bolt_rotation.apply(entry_bolt) - nut_rotation.apply([*params["nut_axis_offset"], face_height])

    def asset(pos, rotation):
        quat = rotation.as_quat()[[3, 0, 1, 2]]
        return SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor(pos, dtype=torch.float32),
            root_quat_w=torch.tensor(np.tile(quat, (count, 1)), dtype=torch.float32),
        ))

    return SimpleNamespace(
        num_envs=count, device="cpu",
        scene={"nut": asset(nut_pos, nut_rotation), "bolt": asset(bolt_pos, bolt_rotation)},
    )


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("bolt_angles", [(0, 0, 0), (0, 0, 125), (20, 30, 40)])
def test_both_faces_use_same_depth_in_bolt_frame(success_check, flipped, bolt_angles):
    check, params = success_check
    assert params["thread_pitch"] * (params["target_thread_turns"] - params["success_threshold_turns"]) == pytest.approx(0.01575)
    env = _env(params, flipped=flipped, bolt_angles=bolt_angles)
    assert check(env, **params).tolist() == [False, False, True, True, False]


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("tilt", [30, 90])
def test_tilted_face_cannot_fake_thread_engagement(success_check, flipped, tilt):
    check, params = success_check
    env = _env(params, flipped=flipped, bolt_angles=(0, 0, 125), tilt=tilt)
    assert not check(env, **params).any()
