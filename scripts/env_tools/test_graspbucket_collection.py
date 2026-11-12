"""Exercise bucket success through the recorder without launching Isaac Sim."""

import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/dexverse"))
from test_reset_exclusions import load_function


def bucket_constants():
    tree = ast.parse((ROOT / "source/dexverse/dexverse/tasks/config/articulation/graspbucket_cfg.py").read_text())
    scope = {"math": math}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith("BUCKET_UPRIGHT_") for t in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "bucket_constants", "exec"), scope)
    return scope


def test_bucket_recording_requires_ten_consecutive_upright_steps():
    scope = {"torch": torch, "resolve_joint_ids": lambda env, cfg: [0],
             "SceneEntityCfg": lambda name: NS(name=name)}
    load_function("terminations.py", "_compare", scope)
    predicate = load_function("terminations.py", "joint_reach_threshold", scope)
    constants = bucket_constants()
    cfg = NS(func=predicate, params={
        "asset_cfg": NS(name="bucket"), "op": "==",
        "threshold": constants["BUCKET_UPRIGHT_ANGLE"], "tol": constants["BUCKET_UPRIGHT_TOLERANCE"],
    })
    path = ROOT / "scripts/record_demos.py"
    tree = ast.parse(path.read_text())
    argument = next(n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "add_argument"
                    and n.value.args and isinstance(n.value.args[0], ast.Constant)
                    and n.value.args[0].value == "--num_success_steps")
    parser = argparse.ArgumentParser()
    exec(compile(ast.Module(body=[argument], type_ignores=[]), str(path), "exec"), {"parser": parser})
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_success")
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    scope.update(args_cli=parser.parse_args([]),
                 _resolve_term_callable=lambda env, cfg: cfg.func)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    env = NS(scene={"bucket": NS(data=NS(joint_pos=torch.zeros(1, 1)))})

    def sweep(angles):
        count = 0
        saved_at = []
        for i, degrees in enumerate(angles):
            env.scene["bucket"].data.joint_pos.fill_(math.radians(degrees))
            count, success = scope["check_success"](env, cfg, count)
            if success:
                saved_at.append(i)
        return saved_at

    assert sweep([-151.2, -90, -70, -64, -56, -48, -40, 0, 40.68]) == []
    assert sweep([-56] * 10) == [9]
    assert sweep([-56] * 9 + [-40] + [-56] * 10) == [19]
