"""Reset clearance regressions without launching Isaac Sim."""

import __future__
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_function(filename, name, scope):
    path = ROOT / "source/dexverse/dexverse/tasks/mdp" / filename
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), scope)
    return scope[name]


@pytest.fixture
def reset():
    def yaw_rotation(roll, pitch, yaw):
        assert not torch.any(roll) and not torch.any(pitch)
        return torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw),
                            torch.zeros_like(yaw), torch.sin(yaw / 2)), dim=-1)

    def identity_times_delta(initial, delta):
        assert torch.all(initial[:, 0] == 1) and not torch.any(initial[:, 1:])
        return delta

    return load_function("resets.py", "reset_root_pose_uniform_excluding", {
        "torch": torch,
        "resolve_env_ids": lambda env, ids: torch.arange(env.num_envs, device=env.device) if ids is None else ids,
        "quat_from_euler_xyz": yaw_rotation,
        "quat_mul": identity_times_delta,
    })


def fixture(device="cpu", count=256):
    origins = torch.zeros(count, 3, device=device)
    origins[:, 0] = torch.arange(count, device=device) * 3
    state = torch.zeros(count, 13, device=device)
    state[:, 1], state[:, 2], state[:, 3] = .1, .718, 1
    writes = {}
    def save_pose(pose, env_ids):
        writes.update(pose=pose, ids=env_ids)
    def save_velocity(velocity, env_ids):
        writes["velocity"] = velocity
    bucket = NS(data=NS(default_root_state=state), cfg=NS(spawn=None),
                write_root_pose_to_sim=save_pose, write_root_velocity_to_sim=save_velocity)
    hand = NS(data=NS(root_pos_w=origins + torch.tensor([-.75, 0, .5], device=device),
                     body_pos_w=origins[:, None] + torch.tensor([[[-.25, 0, .8], [-.055, .011, .8]]], device=device)))
    env = NS(num_envs=count, device=device, scene={"bucket": bucket, "hand": hand})
    env.scene = type("Scene", (dict,), {})(env.scene)
    env.scene.env_origins = origins
    kwargs = dict(asset_cfg=NS(name="bucket"), reference_asset_cfg=NS(name="hand", body_ids=slice(None)),
                  pose_range={"x": [0, 0], "y": [-.1, .1]}, min_xy_distance=.16,
                  use_reference_bodies=True, fail_on_exhaustion=True, max_attempts=100)
    return env, kwargs, writes


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))])
def test_link_clearance_preserves_range_and_selected_environment_ids(reset, device):
    torch.manual_seed(42)
    env, kwargs, writes = fixture(device)
    ids = torch.arange(1, env.num_envs, 2, device=device)
    reset(env, ids, **kwargs)
    pose = writes["pose"]
    assert torch.equal(writes["ids"], ids)
    local = pose[:, :3] - env.scene.env_origins[ids]
    assert float(local[:, 0].abs().max()) < 1e-5
    assert bool(((local[:, 1] >= 0) & (local[:, 1] <= .2)).all())
    distance = (pose[:, None, :2] - env.scene["hand"].data.body_pos_w[ids, :, :2]).norm(dim=-1)
    assert float(distance.min()) >= .16 - 1e-5
    assert not torch.any(writes["velocity"])


def test_impossible_clearance_raises_without_writing_unsafe_pose(reset):
    env, kwargs, writes = fixture(count=2)
    kwargs.update(pose_range={"x": [0, 0], "y": [-.1, -.1]}, max_attempts=3)
    with pytest.raises(RuntimeError, match="No collision-free reset"):
        reset(env, None, **kwargs)
    assert not writes


def test_root_mode_still_clears_reference_root_without_body_data(reset):
    env, kwargs, writes = fixture(count=256)
    kwargs.update(use_reference_bodies=False, min_xy_distance=.10,
                  pose_range={"x": [-.15, .15], "y": [-.2, .2]})
    env.scene["hand"].data.root_pos_w = env.scene.env_origins.clone()
    del env.scene["hand"].data.body_pos_w
    reset(env, None, **kwargs)
    distance = (writes["pose"][:, :2] - env.scene.env_origins[:, :2]).norm(dim=-1)
    assert float(distance.min()) >= .10 - 1e-5


def test_equality_tolerance_rejects_both_sides_of_target():
    compare = load_function("terminations.py", "_compare", {"torch": torch})
    assert compare(torch.tensor([1., 1.9, 2., 2.2, 3.]), 2., "==", .25).tolist() == [False, True, True, True, False]


def test_bucket_config_retains_visible_position_and_yaw_variation_after_rejection(reset):
    path = ROOT / "source/dexverse/dexverse/tasks/config/articulation/graspbucket_cfg.py"
    cfg = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "GraspBucketEnvFloatingDexHandRightCfg")
    ranges = next(node.value for node in cfg.body if isinstance(node, ast.AnnAssign)
                  and node.target.id == "articulation_reset_pose_range")
    ranges = eval(compile(ast.Expression(ranges), str(path), "eval"), {"math": math})
    env, kwargs, writes = fixture(count=512)
    kwargs.update(pose_range=ranges, max_attempts=512)
    torch.manual_seed(42)
    poses = []
    for _ in range(2):
        reset(env, None, **kwargs)
        pose = writes["pose"]
        distance = (pose[:, None, :2] - env.scene["hand"].data.body_pos_w[:, :, :2]).norm(dim=-1)
        assert float(distance.min()) >= .16 - 1e-5
        poses.append(pose.clone())
    assert not torch.equal(poses[0], poses[1])
    xy = torch.cat([p[:, :2] - env.scene.env_origins[:, :2] for p in poses])
    x, y = xy.unbind(-1)
    quats = torch.cat([p[:, 3:7] for p in poses])
    yaw = 2 * torch.atan2(quats[:, 3], quats[:, 0])
    assert float(x.max() - x.min()) > .38
    assert float(x.min()) >= ranges["x"][0] - 1e-5
    assert float(x.max()) <= ranges["x"][1] + 1e-5
    assert float(y.max() - y.min()) > .38
    assert float(y.min()) >= .1 + ranges["y"][0] - 1e-5
    assert float(y.max()) <= .1 + ranges["y"][1] + 1e-5
    assert float(yaw.min()) >= ranges["yaw"][0] - 1e-5
    assert float(yaw.max()) <= ranges["yaw"][1] + 1e-5
    assert float(torch.rad2deg(yaw.max() - yaw.min())) > 55
