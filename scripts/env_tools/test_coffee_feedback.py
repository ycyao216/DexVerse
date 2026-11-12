"""Coffee cue regressions: exact pour conditions, read-only display, reset and USD."""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'source/dexverse/dexverse/tasks/config'


def package(name, path):
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


package('coffee_feedback_test', CONFIG)
package('coffee_feedback_test.bimanual', CONFIG / 'bimanual')
package('coffee_feedback_test.long_horizon', CONFIG / 'long_horizon')
package('coffee_feedback_test.long_horizon.make_coffee', CONFIG / 'long_horizon/make_coffee')
from coffee_feedback_test.long_horizon.make_coffee import checks, feedback
from coffee_feedback_test.long_horizon.check_utils import track


class Scene(dict):
    pass


def sample_env(n=1, *, bottle_tilt=85, mug_tilt=0, xy=0, height=.055):
    mug_q = torch.tensor(Rotation.from_euler('x', mug_tilt, degrees=True).as_quat()[[3, 0, 1, 2]], dtype=torch.float32).repeat(n, 1)
    bottle_q = torch.tensor(Rotation.from_euler('x', bottle_tilt, degrees=True).as_quat()[[3, 0, 1, 2]], dtype=torch.float32).repeat(n, 1)
    mug_p = torch.tensor([[0., 0., .6]]).repeat(n, 1)
    rim = checks._point_above_root(mug_p, mug_q, checks.MUG_RIM_HEIGHT)
    bottle_p = rim + torch.tensor([xy, 0, height]) - checks._point_above_root(torch.zeros(n, 3), bottle_q, checks.MILK_SPOUT_HEIGHT)
    scene = Scene(object=SimpleNamespace(data=SimpleNamespace(root_pos_w=mug_p, root_quat_w=mug_q)),
                  milk=SimpleNamespace(data=SimpleNamespace(root_pos_w=bottle_p, root_quat_w=bottle_q)))
    scene.env_origins = torch.zeros(n, 3)
    return SimpleNamespace(scene=scene, num_envs=n, device='cpu', step_dt=1/60,
                           episode_length_buf=torch.zeros(n, dtype=torch.long), cfg=SimpleNamespace(enable_debug_vis=False))


@pytest.mark.parametrize('values,failed_index', [
    ({'xy': .045}, 0), ({'height': .009}, 1), ({'height': .121}, 1),
    ({'bottle_tilt': 69}, 2), ({'mug_tilt': 21}, 3),
])
def test_stream_requires_every_real_pour_condition(values, failed_index):
    env = sample_env(**values)
    for step in range(90):
        env.episode_length_buf[:] = step
        assert not checks.milk_poured_over_mug(env).any()
    display = feedback.coffee_display_state(env)
    assert not display['flowing'].any()
    assert not display['conditions'][0, failed_index]
    assert display['progress'].item() == 0


def test_hold_progress_completion_and_interruption():
    env = sample_env()
    for step in range(30):
        env.episode_length_buf[:] = step
        assert not checks.milk_poured_over_mug(env).any()
    display = feedback.coffee_display_state(env)
    assert display['flowing'].item()
    assert display['progress'].item() == pytest.approx(.5)
    before = {key: tuple(v.clone() for v in value) if key == 'pour' else value
              for key, value in env._long_horizon_check_memory.items()}
    for _ in range(50):
        feedback.coffee_display_state(env)
    assert all(torch.equal(a, b) for a, b in zip(before['pour'], env._long_horizon_check_memory['pour']))
    env.scene['milk'].data.root_pos_w[:, 0] += .1
    env.episode_length_buf += 1
    checks.milk_poured_over_mug(env)
    assert feedback.coffee_display_state(env)['progress'].item() == 0
    env.scene['milk'].data.root_pos_w[:, 0] -= .1
    for _ in range(60):
        env.episode_length_buf += 1
        poured = checks.milk_poured_over_mug(env)
    assert poured.item()
    flags = {'pour_milk': poured, 'place_on_machine': torch.tensor([False])}
    display = feedback.coffee_display_state(env, flags)
    assert display['stages'].tolist() == [[2, 1, 0]]
    assert display['progress'].item() == 1
    assert not display['flowing'].item()


def test_completed_stages_spill_and_partial_reset():
    env = sample_env(2)
    flags = {'pour_milk': torch.ones(2, dtype=torch.bool), 'place_on_machine': torch.ones(2, dtype=torch.bool),
             'rotate_knob': torch.ones(2, dtype=torch.bool)}
    # A historical lever latch must not show current success after losing the end state.
    display = feedback.coffee_display_state(env, flags, torch.tensor([True, False]))
    assert display['stages'].tolist() == [[2, 2, 2], [2, 2, 1]]
    for step in range(6):
        env.episode_length_buf[:] = step
        track(env, 'spill', torch.tensor([True, False]), .1)
    display = feedback.coffee_display_state(env, flags)
    assert display['status'].tolist() == [4, 2]
    assert display['stages'][0].tolist() == [3, 3, 3]
    env._stage_graph_runtime_cache = {'coffee': SimpleNamespace(latched_flags=flags)}
    feedback.reset_coffee_feedback(env, torch.tensor([0]), 'coffee')
    display = feedback.coffee_display_state(env, env._stage_graph_runtime_cache['coffee'].latched_flags)
    assert display['stages'].tolist() == [[1, 0, 0], [2, 2, 1]]
    assert not display['spilled'].any()


def test_usd_cues_are_visible_nonphysical_and_follow_live_poses():
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics
    from coffee_feedback_test.long_horizon.make_coffee.visualization import CoffeeVisuals, ROOT as VISUAL_ROOT
    env = sample_env()
    env.sim = SimpleNamespace(stage=Usd.Stage.CreateInMemory())
    env.cfg.scene = SimpleNamespace(table=SimpleNamespace(init_state=SimpleNamespace(pos=(0,0,.575)),
                                                          spawn=SimpleNamespace(size=(1.5,1.5,.05))))
    visuals = CoffeeVisuals(env)
    display = feedback.coffee_display_state(env)
    visuals.update(env, display)
    for prim in env.sim.stage.Traverse():
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        if prim.IsA(UsdGeom.Imageable):
            assert UsdGeom.Imageable(prim).ComputePurpose() != 'guide'
    def visible(marker):
        return max(marker.GetScalesAttr().Get()[0]) > 1e-5
    assert visible(visuals.lines[1])
    assert not visible(visuals.surface)
    assert list(visuals.bar.GetProtoIndicesAttr().Get()) == [0] * 30
    target = torch.tensor([list(p) for p in visuals.target.GetPositionsAttr().Get()])
    torch.testing.assert_close(target, display['rim'])
    flags = {'pour_milk': torch.tensor([True])}
    visuals.update(env, feedback.coffee_display_state(env, flags))
    assert not visible(visuals.lines[1])
    assert not visible(visuals.target)
    assert visible(visuals.surface)
    assert list(visuals.bar.GetProtoIndicesAttr().Get()) == [1] * 30
    env.cfg.enable_debug_vis = True
    env._coffee_visuals = visuals
    feedback.reset_coffee_feedback(env, None, 'coffee')
    assert not visible(visuals.surface)
    assert visible(visuals.target)
    assert list(visuals.bar.GetProtoIndicesAttr().Get()) == [0] * 30
