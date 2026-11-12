"""Offline regressions for category installation, selection and baseline views."""
import ast
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'demo_tools'
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parents[1] / 'source/dexverse'))
import demo_layout as layout
import download_collections as download
import download_demos as download_cli
import merge_session_pickles as merge
from organize_verified_demos import action_digest, select_verified_payload


def options(**kwargs):
    return SimpleNamespace(**dict(dict(version='v0', baseline=False, main=False, cross_20=False, all=False,
                                      category=[], task=[], pattern=[]), **kwargs))


def make_pickle(root, task='Dexverse-PushT-v0', category='non_prehensile', content=b'data'):
    path = root / category / task / (task + '.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_versions_install_side_by_side_and_baseline_copies_are_excluded():
    files = ['demonstrations/v0/non_prehensile/Dexverse-PushT-v0/demos.pkl',
             'demonstrations/v1/non_prehensile/Dexverse-PushT-v1/demos.pkl',
             'demonstrations/baseline/Dexverse-PushT-v0.pkl',
             'demonstrations/release_manifest.json']
    entries = download.select_files(files, options(baseline=True, version='all'))
    assert [e['local_path'] for e in entries] == [
        'non_prehensile/Dexverse-PushT-v0/Dexverse-PushT-v0.pkl',
        'non_prehensile/Dexverse-PushT-v1/Dexverse-PushT-v1.pkl']
    assert len(download.select_files(files, options(all=True))) == 1


def test_new_nonbaseline_tasks_are_selectable():
    remote = 'demonstrations/articulation/Dexverse-PushButton-v0/Dexverse-PushButton-v0.pkl'
    assert download.select_files([remote], options(task=['Dexverse-PushButton-v0']))[0]['path'] == remote
    with pytest.raises(ValueError, match='No remote'):
        download.select_files([], options(task=['articulation/Dexverse-PushButton-v0']))


def test_ambiguous_remote_layout_is_rejected():
    files = ['demonstrations/v0/non_prehensile/Dexverse-PushT-v0/demos.pkl',
             'demonstrations/non_prehensile/Dexverse-PushT-v0/Dexverse-PushT-v0.pkl']
    with pytest.raises(ValueError, match='Multiple remote'):
        download.select_files(files, options(all=True))


def test_main_selection_keeps_baseline_and_v1_separate():
    main = 'demonstrations/main/v0/articulation/Dexverse-OpenDoor-v0/shadow_part_001.pkl'
    coffee = 'demonstrations/main/v0/long_horizon/Dexverse-MakeCoffee-v0/shadow_part_001.pkl'
    baseline = 'demonstrations/v0/articulation/Dexverse-OpenDoor-v0/demos.pkl'
    v1 = 'demonstrations/v1/articulation/Dexverse-OpenDoor-v1/demos.pkl'
    files = [main, coffee, baseline, v1, main.replace('.pkl', '.mp4'),
             'demonstrations/main/release_manifest.json']
    assert {e['path'] for e in download.select_files(files, options(main=True))} == {main, coffee}
    assert {e['path'] for e in download.select_files(files, options(baseline=True, version='all'))} == {baseline, v1}
    all_entries = download.select_files(files, options(all=True, version='all'))
    assert len(all_entries) == len({e['local_path'] for e in all_entries}) == 4
    assert layout.local_release_path(main).startswith('main/articulation/')


def test_cross_20_selection_is_separate_from_main_baseline_and_v1():
    cross = 'demonstrations/cross_20/v0/articulation/Dexverse-OpenFaucet-v0/floating_allegro_right/part_001.pkl'
    main = 'demonstrations/main/v0/articulation/Dexverse-OpenFaucet-v0/shadow_part_001.pkl'
    baseline = 'demonstrations/v0/articulation/Dexverse-OpenFaucet-v0/demos.pkl'
    v1 = baseline.replace('v0', 'v1')
    files = [cross, main, baseline, v1, cross.replace('.pkl', '.mp4')]
    assert [e['path'] for e in download.select_files(files, options(cross_20=True))] == [cross]
    assert [e['path'] for e in download.select_files(files, options(main=True))] == [main]
    assert {e['path'] for e in download.select_files(files, options(baseline=True, version='all'))} == {baseline, v1}
    assert {e['path'] for e in download.select_files(files, options(main=True, cross_20=True))} == {main, cross}
    assert len(download.select_files(files, options(all=True, version='all'))) == 4
    assert layout.local_release_path(cross) == 'cross_20/articulation/Dexverse-OpenFaucet-v0/floating_allegro_right/part_001.pkl'


@pytest.mark.parametrize('suffix', [
    'articulation/Dexverse-OpenFaucet-v0/floating_allegro_right/part.pkl',
    'v1/articulation/Dexverse-OpenFaucet-v0/floating_allegro_right/part.pkl',
    'v0/articulation/Dexverse-OpenFaucet-v0/part.pkl',
    'v0/articulation/Dexverse-OpenFaucet-v0/floating_unknown_right/part.pkl',
    'v0/articulation/Dexverse-OpenFaucet-v0/floating_allegro_right/part.mp4',
    'v0/articulation/Dexverse-OpenFaucet-v0/floating_allegro_right/nested/part.pkl'])
def test_cross_20_rejects_noncanonical_paths(suffix):
    with pytest.raises(ValueError):
        layout.local_release_path('demonstrations/cross_20/' + suffix)


def test_cross_20_inventory_is_complete():
    proof = download.cross_20_metadata()
    manifest = json.loads(download.CROSS_20_MANIFEST.read_text())
    assert sum(e['episodes'] for e in proof.values()) == 840
    assert manifest['state_replay'] and manifest['all_episodes_have_T_plus_1_states']
    assert sum(e['episodes'] for e in proof.values()
               if e['task'] == 'Dexverse-StackCube-v0' and e['robot_type'] == 'floating_allegro_right') == 20


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'wrong_robot', 'checksum'])
def test_cross_20_rejects_damaged_manifest(tmp_path, monkeypatch, mutation):
    manifest = json.loads(download.CROSS_20_MANIFEST.read_text())
    if mutation == 'missing':
        manifest['files'].pop(0)
    elif mutation == 'extra':
        manifest['files'][0]['episodes'] += 1
    elif mutation == 'wrong_robot':
        manifest['files'][0]['robot_type'] = 'floating_allegro_right'
    else:
        manifest['files'][0]['sha256'] = 'invalid'
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(download, 'CROSS_20_MANIFEST', path)
    with pytest.raises(ValueError):
        download.cross_20_metadata()


@pytest.mark.parametrize('path', [
    'demonstrations/main/articulation/Dexverse-OpenDoor-v0/demo.pkl',
    'demonstrations/main/v1/articulation/Dexverse-OpenDoor-v0/demo.pkl',
    'demonstrations/main/v0/articulation/Dexverse-OpenDoor-v0/demo.mp4',
    'demonstrations/main/v0/articulation/Dexverse-OpenDoor-v0/nested/demo.pkl',
    'demonstrations/main/v0/articulation/Dexverse-OpenDoor-v0/../demo.pkl'])
def test_main_rejects_noncanonical_paths(path):
    with pytest.raises(ValueError):
        layout.local_release_path(path)


def test_main_inventory_contains_all_approved_state_demos():
    proof = download.main_metadata()
    manifest = json.loads(download.MAIN_MANIFEST.read_text())
    assert len(proof) == 142
    assert len({e['task'] for e in proof.values()}) == 42
    assert sum(e['episodes'] for e in proof.values()) == 2034
    assert manifest['state_replay'] and manifest['all_episodes_have_T_plus_1_states']
    assert sum(e['episodes'] for e in proof.values() if e['task'] == 'Dexverse-OpenFoldablePhone-v0') == 50
    assert not any('huawei' in e['path'].lower() or 'huawei' in e['task'].lower() for e in proof.values())
    for task in ('Dexverse-PickFromClutter-v0', 'Dexverse-InsertPeg-v0'):
        assert sum(e['episodes'] for e in proof.values() if e['task'] == task) == 50


def test_main_train_and_validation_are_disjoint_and_extra_object_is_excluded():
    proof = download.main_metadata()
    train = download.select_files(list(proof), options(main=True))
    validation = [e for e in download.select_files(list(proof), options(all=True))
                  if '/validation/' in e['local_path']]
    assert len(train) == 122 and len(validation) == 20
    assert {e['path'] for e in train}.isdisjoint(e['path'] for e in validation)
    assert sum(proof[e['path']]['episodes'] for e in train) == 2014
    relocate = [e for e in proof.values() if e['task'] == 'Dexverse-RelocateObject-v0']
    train_objects = {e['object_id'] for e in relocate if e['split'] == 'train'}
    validation_objects = {e['object_id'] for e in relocate if e['split'] == 'validation'}
    assert len(relocate) == 70 and len(validation_objects) == 20
    assert train_objects.isdisjoint(validation_objects)
    assert not any(e['original_demo_number'] == 71 for e in relocate)
    assert all('/train/' in e['local_path'] for e in train if 'RelocateObject' in e['path'])
    assert all('/validation/' in e['local_path'] for e in validation)


def test_foldable_phone_uses_existing_canonical_identity_without_changing_episodes():
    from dexverse.recording_compat import normalize_recording_identity
    original = {'task': 'Dexverse-OpenHuaweiPhone-v0', 'env_name': 'Dexverse-OpenHuaweiPhone-v0',
                'episodes': [{'actions': np.arange(8), 'states': [{'frame': i} for i in range(9)]}]}
    result = normalize_recording_identity(original)
    assert result['task'] == result['env_name'] == 'Dexverse-OpenFoldablePhone-v0'
    assert result['episodes'] is original['episodes']
    categories = layout.task_categories()
    assert categories['OpenFoldablePhone'] == 'articulation'
    assert 'OpenHuaweiPhone' not in categories and 'OpenPhone' not in categories


@pytest.mark.parametrize('incomplete', [False, True])
@pytest.mark.parametrize('collection', ['main', 'cross_20'])
def test_collection_cli_pins_revision_verifies_downloads_and_preserves_existing_data(tmp_path, monkeypatch, incomplete, collection):
    import huggingface_hub
    remote = ('demonstrations/main/v0/articulation/Dexverse-OpenDoor-v0/shadow_part_001.pkl'
              if collection == 'main' else 'demonstrations/cross_20/v0/articulation/Dexverse-OpenFaucet-v0/floating_allegro_right/part_001.pkl')
    cached = tmp_path / 'cached'; cached.write_bytes(b'approved state demo')
    proof = {remote: {'bytes': cached.stat().st_size, 'sha256': layout.sha256(cached)}}
    dest = tmp_path / 'demos'
    old = [make_pickle(dest, 'Dexverse-OpenDoor-v0', 'articulation', b'old v0'),
           make_pickle(dest, 'Dexverse-OpenDoor-v1', 'articulation', b'old v1'),
           make_pickle(dest, 'Dexverse-OpenDoor-v0', 'baseline', b'baseline copy'),
           make_pickle(dest / 'main', 'Dexverse-OpenFaucet-v0', 'articulation', b'other main data')]
    before = {p: p.read_bytes() for p in old}
    downloads = []
    class Api:
        def __init__(self, **kwargs):
            assert kwargs['token'] is False  # Public main downloads need no login.
        def repo_info(self, **kwargs):
            return SimpleNamespace(sha='pinned-commit')
        def list_repo_files(self, **kwargs):
            assert kwargs['revision'] == 'pinned-commit'
            return [] if incomplete else [remote]
    def fetch(**kwargs):
        assert kwargs['revision'] == 'pinned-commit'
        assert kwargs['token'] is False
        downloads.append(kwargs['filename'])
        return str(cached)
    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fetch)
    monkeypatch.setattr(download, collection + '_metadata', lambda: proof)
    monkeypatch.setattr(sys, 'argv', ['download_demos.py', '--' + collection, '--dest', str(dest)])
    if incomplete:
        with pytest.raises(SystemExit) as exc:
            download_cli.main()
        assert exc.value.code == 1
        assert downloads == []
        assert not (dest / layout.local_release_path(remote)).exists()
    else:
        assert download_cli.main() == 0
        assert download_cli.main() == 0  # Verified files are reused without another download.
        assert downloads == [remote]
        assert (dest / layout.local_release_path(remote)).read_bytes() == cached.read_bytes()
    assert {p: p.read_bytes() for p in old} == before


@pytest.mark.parametrize('path', [
    'demonstrations/v1/articulation/Dexverse-PushButton-v0/demos.pkl',
    'demonstrations/../escape.pkl', '/demonstrations/a/b/c.pkl',
    'demonstrations/articulation/../../escape.pkl'])
def test_remote_path_validation(path):
    with pytest.raises(ValueError):
        layout.local_release_path(path)


def test_relative_baseline_links_survive_move_and_do_not_duplicate_merger_inputs(tmp_path):
    root = tmp_path / 'demos'
    source = make_pickle(root)
    entries = ['non_prehensile/Dexverse-PushT-v0', 'articulation/Dexverse-OpenDoor-v0']
    result = layout.sync_baseline_links(root, entries)
    link = root / result['links'][0]
    assert link.is_symlink() and not os.path.isabs(os.readlink(link))
    assert result['missing'] == entries[1:]
    assert layout.sync_baseline_links(root, entries) == result
    assert merge._discover_pickles(root, [], True) == [source.resolve()]
    renamed = tmp_path / 'moved'
    root.rename(renamed)
    assert (renamed / result['links'][0]).read_bytes() == b'data'


def test_baseline_collision_preserves_existing_file(tmp_path):
    root = tmp_path / 'demos'
    source = make_pickle(root)
    baseline = root / 'baseline'
    baseline.mkdir()
    conflict = baseline / source.name
    conflict.write_bytes(b'keep me')
    with pytest.raises(FileExistsError):
        layout.sync_baseline_links(root, ['non_prehensile/Dexverse-PushT-v0'])
    assert conflict.read_bytes() == b'keep me'


def test_category_symlink_cannot_write_outside_destination(tmp_path):
    dest = tmp_path / 'dest'; dest.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    (dest / 'articulation').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        layout.contained_path(dest, 'articulation/Dexverse-PushButton-v0/demo.pkl')
    assert list(outside.iterdir()) == []


def test_download_install_reuses_verified_file_and_rejects_changed_data(tmp_path):
    remote = 'demonstrations/v0/non_prehensile/Dexverse-PushT-v0/demos.pkl'
    cached = tmp_path / 'cached'; cached.write_bytes(b'remote demo')
    entry = {'path': remote, 'local_path': layout.local_release_path(remote)}
    proof = {remote: {'bytes': len(b'remote demo'), 'sha256': layout.sha256(cached)}}
    dest = tmp_path / 'dest'
    fetches = []
    def fetch(path):
        fetches.append(path)
        return cached
    download.fetch_selected([entry], dest, fetch, proof)
    download.fetch_selected([entry], dest, fetch, proof)
    assert fetches == [remote]
    target = dest / entry['local_path']; target.write_bytes(b'local work')
    with pytest.raises(FileExistsError, match='Different local'):
        download.fetch_selected([entry], dest, fetch, proof)
    assert target.read_bytes() == b'local work'
    assert fetches == [remote]


def test_corrupt_download_is_not_installed(tmp_path):
    source = tmp_path / 'cached'; source.write_bytes(b'bad')
    remote = 'demonstrations/articulation/Dexverse-PushButton-v0/demos.pkl'
    entry = {'path': remote, 'local_path': layout.local_release_path(remote)}
    proof = {remote: {'bytes': 3, 'sha256': hashlib.sha256(b'yes').hexdigest()}}
    with pytest.raises(ValueError, match='checksum'):
        download.fetch_selected([entry], tmp_path / 'dest', lambda _: source, proof)
    assert not (tmp_path / 'dest' / entry['local_path']).exists()


def test_converter_discovers_category_groups_once_through_baseline_links(tmp_path):
    make_pickle(tmp_path)
    make_pickle(tmp_path, 'Dexverse-PushButton-v0', 'articulation')
    layout.sync_baseline_links(tmp_path, ['non_prehensile/Dexverse-PushT-v0'])
    # Both replay entry points use this simulator-free discovery implementation.
    from demo_selection import discover_groups
    groups, warnings = discover_groups(tmp_path, all_tasks=True, legacy_collections=True)
    assert not warnings
    assert len(groups) == 2
    assert all(len(group.pickles) == 1 for group in groups)
    assert all(group.anchor_dir.name != 'baseline' for group in groups)


@pytest.mark.parametrize('fmt,prefix', [('dexbench_trajectory', 'Dexbench-'), ('dexverse_trajectory', 'Dexverse-')])
def test_converter_loads_original_state_recordings_without_changing_disk_data(tmp_path, fmt, prefix):
    tree = ast.parse((TOOLS / 'create_demo_files_sequential.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_load_trajectory_pickle')
    namespace = {'pickle': pickle, '_SINGLE_FORMAT': 'dexverse_trajectory', '_BATCH_FORMAT': 'dexverse_trajectory_batch'}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<loader>', 'exec'), namespace)
    payload = {'format': fmt, 'env_name': prefix + 'PickFromClutter-v0',
               'task': prefix + 'PickFromClutter-v0', 'robot_type': 'floating_shadow_right',
               'episodes': [{'actions': [[1, 2]], 'states': [{'x': 0}, {'x': 1}],
                             'multi_assets': {'object_1': {'index': 3}}}]}
    path = tmp_path / 'original.pkl'; path.write_bytes(pickle.dumps(payload))
    before = path.read_bytes()
    loaded = namespace['_load_trajectory_pickle'](str(path))
    assert loaded['env_name'] == loaded['task'] == 'Dexverse-PickFromClutter-v0'
    assert loaded['format'] == 'dexverse_trajectory'
    assert loaded['episodes'] == payload['episodes']
    assert loaded['robot_type'] == payload['robot_type']
    assert path.read_bytes() == before


def verified_fixture():
    payload = {'format': 'dexverse_trajectory', 'episodes': []}
    rows = []
    for i in range(51):
        actions = np.full((12, 2), i, dtype=np.float32)
        payload['episodes'].append({'episode_index': i, 'episode_name': f'source_{i}', 'actions': actions,
                                    'initial_state': {'joint': np.array([i])}, 'goal_pose': [i],
                                    'states': [{'frame': j} for j in range(13)]})
        if i:
            rows.append({'prepared_index': i, 'actions_sha256': action_digest(actions), 'actual_steps': 12,
                         'success': True, 'complete': True, 'max_success_streak': 10})
    return payload, rows


def test_selection_keeps_exact_actions_states_goals_and_excludes_unselected_episode():
    payload, rows = verified_fixture()
    result = select_verified_payload(payload, rows)
    assert len(payload['episodes']) == 51 and payload['episodes'][1]['episode_index'] == 1
    assert result['num_episodes'] == 50
    assert result['episodes'][0]['episode_index'] == 0
    for selected, original in zip(result['episodes'], payload['episodes'][1:]):
        assert np.array_equal(selected['actions'], original['actions'])
        assert selected['states'] == original['states']
        assert selected['goal_pose'] == original['goal_pose']
        assert np.array_equal(selected['initial_state']['joint'], original['initial_state']['joint'])


@pytest.mark.parametrize('field,value', [('success', False), ('complete', False), ('max_success_streak', 9), ('actual_steps', 11)])
def test_failed_or_mismatched_episode_cannot_be_promoted(field, value):
    payload, rows = verified_fixture()
    rows[0][field] = value
    with pytest.raises(ValueError):
        select_verified_payload(payload, rows)


def test_duplicate_selection_cannot_be_promoted():
    payload, rows = verified_fixture()
    rows[1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match='distinct'):
        select_verified_payload(payload, rows)
