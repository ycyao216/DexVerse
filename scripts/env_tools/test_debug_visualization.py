"""CPU regressions for unified debug controls without launching Isaac Sim/XR."""

import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/dexverse"))
from dexverse.teleop_utils.debug_visualization import (  # noqa: E402
    add_debug_visualization_args, configure_v1_debug_visualization,
)


@pytest.mark.parametrize("teleop,debug,rgb", [(True, True, True), (False, None, False)])
def test_shared_cli_defaults_and_switches(teleop, debug, rgb):
    parser = argparse.ArgumentParser()
    add_debug_visualization_args(parser, teleop=teleop)
    args = parser.parse_args([])
    assert args.enable_debug_vis is debug and args.cues_in_rgb is rgb
    args = parser.parse_args(["--no-enable_debug_vis", "--cues_in_rgb"])
    assert args.enable_debug_vis is False and args.cues_in_rgb is True
    args = parser.parse_args(["--enable_debug_vis", "--no-cues_in_rgb"])
    assert args.enable_debug_vis is True and args.cues_in_rgb is False


def _config(version, debug):
    package = "dexverse.baseline_v1" if version == 1 else "dexverse.tasks"
    cls = type("Config", (), {"__module__": f"{package}.config.test"})
    cfg = cls()
    cfg.enable_debug_vis = debug
    cfg.teleop_devices = SimpleNamespace(devices={"handtracking": SimpleNamespace(
        retargeters=[SimpleNamespace(task_version=version, debug_vis=True)]
    )})
    return cfg


@pytest.mark.parametrize("debug,rgb", [(False, False), (False, True), (True, False), (True, True)])
def test_v1_retarg_and_camera_policy_are_independent(monkeypatch, debug, rgb):
    calls = []
    monkeypatch.setitem(sys.modules, "dexverse.baseline_v1.visual_purpose",
                        SimpleNamespace(set_camera_hiding_enabled=calls.append))
    cfg = _config(1, debug)
    configure_v1_debug_visualization(cfg, cues_in_rgb=rgb)
    assert calls == [not rgb]
    rtg = cfg.teleop_devices.devices["handtracking"].retargeters[0]
    assert rtg.task_version == 1 and rtg.debug_vis is debug


def test_v0_rendering_and_retarg_untouched(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "dexverse.baseline_v1.visual_purpose",
                        SimpleNamespace(set_camera_hiding_enabled=calls.append))
    cfg = _config(0, False)
    configure_v1_debug_visualization(cfg, cues_in_rgb=True)
    assert not calls
    assert cfg.teleop_devices.devices["handtracking"].retargeters[0].debug_vis is True


def _method(relative_path, class_name, method_name, namespace):
    path = ROOT / relative_path
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    # Avoid evaluating simulator type annotations.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


def test_disabled_hand_visualizers_do_not_process_tracking_or_show_markers():
    for method in ("_visualize_hand_keypoints", "_visualize_canonical_hand_keypoints", "_visualize_wrist_poses"):
        call = _method("source/dexverse/dexverse/devices/retargeters/simple_relative_retargeting.py",
                       "SimpleRelativeRetargeter", method, {})
        rtg = SimpleNamespace(_debug_visualization_enabled=False)
        call(rtg, None) if method != "_visualize_wrist_poses" else call(rtg)


def test_zero_agent_hydra_debug_flag_is_consumed_before_config_construction():
    path = ROOT / "scripts/zero_agent.py"
    tree = ast.parse(path.read_text())
    loop = next(node for node in tree.body if isinstance(node, ast.For)
                and ast.unparse(node.target) == "raw_arg")
    for explicit, override, expected in ((None, "true", True), (None, "false", False), (False, "true", False)):
        namespace = {
            "hydra_args": [f"env.enable_debug_vis={override}", "env.seed=1000"],
            "remaining_hydra_args": [], "args_cli": SimpleNamespace(enable_debug_vis=explicit),
            "parser": argparse.ArgumentParser(),
        }
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), namespace)
        assert namespace["args_cli"].enable_debug_vis is expected
        assert namespace["remaining_hydra_args"] == ["env.seed=1000"]


@pytest.mark.parametrize("debug", [False, True])
def test_command_goal_remains_visible_when_current_pose_debug_is_off(debug):
    class Marker:
        def __init__(self, cfg):
            self.visible = False
        def set_visibility(self, visible):
            self.visible = visible

    hidden = []
    method = _method("source/dexverse/dexverse/baseline_v1/mdp/commands/pose_commands.py",
                     "ObjectUniformPoseCommand", "_set_debug_vis_impl",
                     {"VisualizationMarkers": Marker, "hide_marker_from_cameras": hidden.append})
    command = SimpleNamespace(_env=SimpleNamespace(cfg=SimpleNamespace(enable_debug_vis=debug)),
                              cfg=SimpleNamespace(goal_pose_visualizer_cfg=None, curr_pose_visualizer_cfg=None))
    method(command, True)
    assert command.goal_visualizer.visible
    assert (command.curr_visualizer is not None) is debug
    assert command.goal_visualizer not in hidden
    assert len(hidden) == int(debug)
    method(command, False)
    assert not command.goal_visualizer.visible


def test_scissors_debug_cues_follow_shared_camera_policy():
    path = ROOT / "source/dexverse/dexverse/baseline_v1/config/articulation/cut_strip_scissors_cfg.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "configure_debug_vis")
    flags = [value.value for node in ast.walk(method) if isinstance(node, ast.Dict)
             for key, value in zip(node.keys, node.values)
             if isinstance(key, ast.Constant) and key.value == "hide_from_cameras"]
    assert flags == [True, True]


def test_camera_hiding_switch_can_be_reset_between_launch_configurations():
    path = ROOT / "source/dexverse/dexverse/baseline_v1/visual_purpose.py"
    spec = importlib.util.spec_from_file_location("isolated_visual_purpose", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.set_camera_hiding_enabled(False)
    assert module.CAMERA_HIDING_ENABLED is False
    module.set_camera_hiding_enabled(True)
    assert module.CAMERA_HIDING_ENABLED is True


def test_optional_hand_cues_are_skipped_when_not_created():
    # Canonical keypoints and wrist frames default off; their markers are None
    # and the visualizers must return before touching tracking data.
    for method, attr in (("_visualize_canonical_hand_keypoints", "_canonical_markers"),
                         ("_visualize_wrist_poses", "_wrist_markers")):
        call = _method("source/dexverse/dexverse/devices/retargeters/simple_relative_retargeting.py",
                       "SimpleRelativeRetargeter", method, {})
        rtg = SimpleNamespace(_debug_visualization_enabled=True, **{attr: None})
        call(rtg, None) if method != "_visualize_wrist_poses" else call(rtg)


def test_optional_hand_cue_flags_default_off():
    path = ROOT / "source/dexverse/dexverse/devices/retargeters/simple_relative_retargeting.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "SimpleRelativeRetargeterCfg")
    defaults = {node.target.id: node.value.value for node in cls.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant)}
    assert defaults["show_canonical_keypoints"] is False
    assert defaults["show_wrist_frames"] is False
