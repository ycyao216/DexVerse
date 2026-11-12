"""Simulator-free regressions for the baking checklist and snapshot ownership."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'source/dexverse/dexverse/tasks/config/long_horizon/oven_bake_salmon/feedback.py'
spec = importlib.util.spec_from_file_location('baking_feedback_test', PATH)
feedback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback)


def test_seasonings_are_unordered_and_loading_waits_for_both():
    env = SimpleNamespace(num_envs=3, device='cpu')
    flags = {'salt_poured': torch.tensor([False, False, True]),
             'pepper_poured': torch.tensor([False, True, True])}
    assert feedback.display_state(env, flags, torch.zeros(3, dtype=torch.bool)).tolist() == [
        [1, 1, 0, 0, 0], [1, 2, 0, 0, 0], [2, 2, 1, 0, 0]]


def test_historical_knob_turn_does_not_display_live_success():
    env = SimpleNamespace(num_envs=2, device='cpu')
    flags = {name: torch.ones(2, dtype=torch.bool) for name in feedback.STAGES}
    assert feedback.display_state(env, flags, torch.tensor([True, False])).tolist() == [
        [2, 2, 2, 2, 2], [2, 2, 2, 2, 1]]


def test_saved_history_does_not_change_on_later_reset():
    flags = {'salt_poured': torch.tensor([True, False]), 'pepper_poured': torch.tensor([False, True])}
    env = SimpleNamespace(episode_length_buf=torch.tensor([12, 45]),
                          _stage_graph_runtime_cache={feedback.TASK_KEY: SimpleNamespace(latched_flags=flags)})
    snapshot = feedback.capture_baking_state(env, 1)
    env.episode_length_buf.zero_()
    for value in flags.values():
        value.zero_()
    assert snapshot['episode_step'].item() == 45
    assert snapshot['stages']['pepper_poured'].item()
    assert not snapshot['stages']['salt_poured'].item()


def test_saving_before_first_predicate_evaluation_is_valid():
    snapshot = feedback.capture_baking_state(SimpleNamespace(episode_length_buf=torch.tensor([0])), 0)
    assert snapshot['stages'] == {} and snapshot['episode_step'].item() == 0
