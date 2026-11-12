"""Startup STOP must not block construction or disable later STOP handling."""

import weakref

import pytest

from dexverse.teleop_utils.simulation_startup import guard_simulation_startup


def simulation_type():
    class Simulation:
        def __init__(self, *, playing=False, fail=False):
            self.events = []
            self.ready = False
            if playing:
                # Isaac Lab's timeline subscription passes a weak proxy.
                type(self)._app_control_on_stop_handle_fn(weakref.proxy(self), "constructor-stop")
            if fail:
                raise ValueError("scene failed")
            self.ready = True

        def _app_control_on_stop_handle_fn(self, event):
            if not self.ready:
                raise RuntimeError("waiting for PLAY before the scene exists")
            self.events.append(event)
            return "handled"

    return Simulation


@pytest.mark.parametrize("playing", [False, True])
def test_early_play_loads_scene_and_preserves_later_stop(playing):
    simulation = simulation_type()
    original_init = simulation.__init__
    original_stop = simulation._app_control_on_stop_handle_fn
    if playing:
        with pytest.raises(RuntimeError, match="waiting for PLAY"):
            simulation(playing=True)
    with guard_simulation_startup(simulation):
        sim = simulation(playing=playing)
        assert sim.ready
        assert sim._app_control_on_stop_handle_fn("reset-stop") == "handled"
    assert simulation.__init__ is original_init
    assert simulation._app_control_on_stop_handle_fn is original_stop
    assert sim._app_control_on_stop_handle_fn("user-stop") == "handled"
    assert sim.events == ["reset-stop", "user-stop"]


def test_failed_construction_restores_original_handlers():
    simulation = simulation_type()
    original_init = simulation.__init__
    original_stop = simulation._app_control_on_stop_handle_fn
    sim = simulation.__new__(simulation)
    with pytest.raises(ValueError, match="scene failed"):
        with guard_simulation_startup(simulation):
            sim.__init__(playing=True, fail=True)
    assert not sim._dexverse_initializing_simulation
    assert simulation.__init__ is original_init
    assert simulation._app_control_on_stop_handle_fn is original_stop
