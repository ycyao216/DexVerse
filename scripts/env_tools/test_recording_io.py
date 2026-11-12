"""A failed recording checkpoint must leave the previous checkpoint recoverable."""

import pickle

import pytest

from dexverse.teleop_utils import recording_io


def test_checkpoint_round_trip_and_atomic_replacement(tmp_path):
    path = tmp_path / "session" / "demos.pkl"
    recording_io.write_recording(path, {"episodes": [1]})
    recording_io.write_recording(path, {"episodes": [1, 2]})
    assert pickle.loads(path.read_bytes()) == {"episodes": [1, 2]}
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["serialize", "sync", "replace"])
def test_failed_checkpoint_preserves_previous_data_and_cleans_temporary_file(tmp_path, monkeypatch, failure):
    path = tmp_path / "demos.pkl"
    recording_io.write_recording(path, {"episodes": [1]})
    before = path.read_bytes()

    def fail(*args, **kwargs):
        if failure == "serialize":
            args[1].write(b"partial pickle")
        raise OSError("checkpoint interrupted")

    target, name = {
        "serialize": (recording_io.pickle, "dump"),
        "sync": (recording_io.os, "fsync"),
        "replace": (recording_io.os, "replace"),
    }[failure]
    monkeypatch.setattr(target, name, fail)
    with pytest.raises(OSError, match="checkpoint interrupted"):
        recording_io.write_recording(path, {"episodes": [1, 2]})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
