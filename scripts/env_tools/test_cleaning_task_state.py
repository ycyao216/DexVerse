"""The cleaning recorder owns snapshots and restores only the selected environment."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch

PATH = Path(__file__).resolve().parents[2] / 'source/dexverse/dexverse/tasks/config/long_horizon/trash_drawer_sort_simple/trash_drawer_sort_simple_cfg.py'


def test_cleaning_history_restores_subset_visibility_latch_and_target_without_touching_other_env():
    # Load the actual state hooks without launching Isaac Sim; stub physics I/O.
    names = {'capture_cleaning_state', 'restore_cleaning_state'}
    nodes = [n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    visibility = []
    class Trash:
        def __init__(self):
            self.data = SimpleNamespace(joint_pos=torch.zeros(2, 2), joint_pos_target=torch.tensor([[-.1, 0.], [-1.2, 0.]]))
        def find_joints(self, pattern):
            return [0], ['lid']
        def set_joint_position_target(self, target, *, joint_ids, env_ids):
            self.data.joint_pos_target[env_ids[:, None], joint_ids] = target
    trash = Trash()
    objects = {name: SimpleNamespace(set_visibility=lambda visible, env_ids, name=name: visibility.append((name, visible, env_ids.tolist())))
               for name in ('headphone', 'can')}
    env = SimpleNamespace(device='cpu', num_envs=2, scene={**objects, 'trash_can': trash},
                          _active_drawer_objects=torch.tensor([[True], [False]]),
                          _active_trash_objects=torch.tensor([[False], [True]]),
                          _trash_can_latched=torch.tensor([True, False]),
                          _trash_can_prev_q=torch.tensor([-.1, -.8]),
                          _trash_can_cooldown_s=torch.tensor([0., .15]), episode_length_buf=torch.tensor([0, 30]))
    scope = {'torch': torch, 'TRASH_CAN_LID_JOINT_NAMES': 'lid', 'ALL_TABLE_CLEANING_NAMES': ('headphone', 'can'),
             '_ensure_active_object_state': lambda env: None, '_ensure_trash_can_latch_state': lambda env: None,
             '_apply_trash_can_latch_state': lambda *a: None, '_install_active_object_metadata_getter': lambda env: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), scope)
    snapshot = scope['capture_cleaning_state'](env, 1)
    for name in snapshot['buffers']:
        getattr(env, name)[1] = 0
    trash.data.joint_pos_target[1] = 0
    assert snapshot['buffers']['_active_trash_objects'].item()
    assert snapshot['lid_target'].item() == torch.tensor(-1.2).item()
    other_env = {name: getattr(env, name)[0].clone() for name in snapshot['buffers']}
    scope['restore_cleaning_state'](env, snapshot, 1)
    for name, saved in snapshot['buffers'].items():
        assert torch.equal(getattr(env, name)[1], saved)
        assert torch.equal(getattr(env, name)[0], other_env[name])
    assert trash.data.joint_pos_target.tolist() == [[torch.tensor(-.1).item(), 0.], [torch.tensor(-1.2).item(), 0.]]
    assert visibility == [('can', True, [1])]
