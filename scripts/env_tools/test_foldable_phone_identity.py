"""Renamed task registration and lossless legacy demonstration compatibility."""

import ast
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/dexverse"))
sys.path.insert(0, str(ROOT / "scripts/demo_tools"))
from dexverse.benchmark import task_identity, validate_replay_identity
from dexverse.recording_compat import (
    FOLDABLE_PHONE_TASK as TASK,
    RETIRED_RECORDING_TASKS,
    normalize_recording_identity,
)
from demo_selection import discover_groups, validate_selected_identity
import merge_demos
import merge_session_pickles

OLD = "Dexverse-OpenHuaweiPhone-v0"


def session(task=OLD):
    return {
        "format": "dexverse_trajectory", "task": task, "env_name": task,
        "robot_type": "floating_shadow_bimanual", "num_episodes": 1,
        "episodes": [{"episode_index": 0, "success": True,
                      "actions": np.arange(8, dtype=np.float32).reshape(2, 4),
                      "initial_state": {"articulation": np.arange(2, dtype=np.float32)},
                      "states": [{}, {}, {}]}],
    }


def load_script(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_helper(name, **namespace):
    path = ROOT / "scripts/demo_tools/create_demo_files_sequential.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_only_generic_task_is_registered_and_listed():
    config_root = ROOT / "source/dexverse/dexverse/tasks/config"
    registrations = []
    for path in config_root.rglob("__init__.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register_env":
                registrations.append(tuple(ast.literal_eval(arg) for arg in node.args[1:]))
    phone = [r for r in registrations if "Phone" in r[0] and "Open" in r[0]]
    assert phone == [(TASK, "open_foldable_phone_cfg", "OpenFoldablePhoneEnvFloatingDexHandRightCfg")]
    assert not set(RETIRED_RECORDING_TASKS) & {r[0] for r in registrations}
    assert not (config_root / "articulation/open_huawei_phone_cfg.py").exists()
    assert task_identity(TASK)["task_version"] == 0
    for old in RETIRED_RECORDING_TASKS:
        with pytest.raises(ValueError, match="Unknown canonical"):
            task_identity(old)


def test_phone_config_preserves_original_physics_and_state_bindings():
    from dexverse.assets.foldable_phone import (
        FOLDABLE_PHONE_USD_PATH, FOLDABLE_PHONE_LEFT_HINGE, FOLDABLE_PHONE_RIGHT_HINGE,
    )
    original = subprocess.run(
        ["git", "show", "30cc673:source/dexverse/dexverse/tasks/config/articulation/open_huawei_phone_cfg.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if original.returncode:
        pytest.skip("Official source snapshot is unavailable outside the repository")
    current = (ROOT / "source/dexverse/dexverse/tasks/config/articulation/open_foldable_phone_cfg.py").read_text()
    bindings = {
        "HUAWEI_PHONE_USD_PATH": FOLDABLE_PHONE_USD_PATH,
        "FOLDABLE_PHONE_USD_PATH": FOLDABLE_PHONE_USD_PATH,
        "FOLDABLE_PHONE_LEFT_HINGE": FOLDABLE_PHONE_LEFT_HINGE,
        "FOLDABLE_PHONE_RIGHT_HINGE": FOLDABLE_PHONE_RIGHT_HINGE,
    }

    class ResolveBindings(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.Constant(bindings[node.id]) if node.id in bindings else node

    def config_body(source):
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
        cls.name = "RenamedTask"
        if isinstance(cls.body[0], ast.Expr) and isinstance(cls.body[0].value, ast.Constant):
            cls.body.pop(0)  # The descriptive class docstring changed with the name.
        return ast.dump(ResolveBindings().visit(cls), include_attributes=False)

    assert config_body(current) == config_body(original.stdout)


@pytest.mark.parametrize("old", list(RETIRED_RECORDING_TASKS) + ["dexverse.tasks:" + OLD])
def test_legacy_metadata_normalization_preserves_episode_objects_and_provenance(old):
    original = session(old)
    result = normalize_recording_identity(original)
    assert result["task"] == result["env_name"] == TASK
    assert result["recorded_task_identity"] == {"task": old, "env_name": old}
    assert result["episodes"] is original["episodes"]
    assert original["task"] == old
    assert normalize_recording_identity(result) is result
    validate_selected_identity(original, TASK)
    validate_replay_identity(original, TASK)


def test_rename_does_not_bypass_identity_or_physics_revision_checks():
    original = session()
    original["env_name"] = "Dexverse-OpenLaptop-v0"
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_recording_identity(original)
    original = session()
    original["benchmark_revision"] = "incompatible-physics"
    with pytest.raises(ValueError, match="does not match"):
        validate_replay_identity(original, TASK)
    unrelated = session("Dexverse-OpenLaptop-v0")
    assert normalize_recording_identity(unrelated) is unrelated
    with pytest.raises(ValueError, match="curated pickle"):
        validate_selected_identity(unrelated, TASK)


def test_replay_loader_and_both_mergers_accept_old_and_new_without_editing_sources(tmp_path):
    paths = [tmp_path / f"{task}__session.pkl" for task in (OLD, TASK)]
    originals = [pickle.dumps(session(task), protocol=4) for task in (OLD, TASK)]
    for path, raw in zip(paths, originals):
        path.write_bytes(raw)
    loader = replay_helper("_load_trajectory_pickle", pickle=pickle,
                           _SINGLE_FORMAT="dexverse_trajectory", _BATCH_FORMAT="dexverse_trajectory_batch",
                           normalize_recording_identity=normalize_recording_identity)
    assert loader(str(paths[0]))["task"] == TASK
    groups, _ = discover_groups(tmp_path, files=[str(paths[0])])
    assert groups[0].pickles == [paths[0]] and groups[0].output_stem == TASK + "__session"
    batch = merge_demos.merge_trajectory_pickles([str(p) for p in paths], str(tmp_path / "batch.pkl"))
    assert batch["task"] == batch["env_name"] == TASK
    np.testing.assert_array_equal(batch["actions"][:, 0], session()["episodes"][0]["actions"])
    output, count, status = merge_session_pickles._merge_one_task(tmp_path, paths, OLD + ".pkl", False)
    assert (output.name, count, status) == (TASK + ".pkl", 2, "written")
    merged = pickle.loads(output.read_bytes())
    assert merged["task"] == merged["env_name"] == TASK
    assert merge_session_pickles._looks_like_merged_output(output, OLD + ".pkl")
    # Rebuilding does not accidentally include the previous canonical output.
    _, count, _ = merge_session_pickles._merge_one_task(tmp_path, paths + [output], OLD + ".pkl", True)
    assert count == 2
    for path, raw in zip(paths, originals):
        assert path.read_bytes() == raw




def test_training_exports_use_current_task_name(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts/dp3"))
    diffusion = load_script("scripts/diffusion/build_dataset.py")
    dp3 = load_script("scripts/dp3/convert_demos_to_dp3.py")
    source = tmp_path / "old.h5"
    actions = np.arange(8, dtype=np.float32).reshape(2, 4)
    with h5py.File(source, "w") as hf:
        hf.attrs["task"] = OLD
        hf.attrs["term_order"] = json.dumps({})
        demo = hf.create_group("data/demo_0")
        demo.attrs["num_samples"] = 2
        demo.create_dataset("actions", data=actions)
        for group in set(diffusion.REQUIRED_GROUPS) | {"proprio"}:
            demo.create_dataset(f"obs/{group}/test", data=np.ones((2, 3), dtype=np.float32))
        demo.create_dataset("obs/pointcloud/camera_point_cloud_w", data=np.ones((2, 4, 3), dtype=np.float32))
    before = source.read_bytes()
    diffusion.build_dataset(source, tmp_path / "diffusion.h5", task=None, pkl_file=None)
    dp3.convert(str(source), str(tmp_path / "dp3.h5"), num_points=4)
    with h5py.File(tmp_path / "diffusion.h5") as hf:
        assert json.loads(hf.attrs["meta"])["task"] == TASK
        np.testing.assert_array_equal(hf["data/demo_0/actions"][:], actions)
    with h5py.File(tmp_path / "dp3.h5") as hf:
        assert hf.attrs["task"] == TASK
        np.testing.assert_array_equal(hf["data/episode_0/action"][:], actions)
    assert source.read_bytes() == before
