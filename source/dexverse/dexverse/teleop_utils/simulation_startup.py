"""Keep an early GUI/XR timeline PLAY from blocking simulation construction."""

from contextlib import contextmanager
from functools import wraps
import logging


logger = logging.getLogger(__name__)


@contextmanager
def guard_simulation_startup(simulation_context_type=None):
    """Ignore STOP callbacks only while the SimulationContext constructor runs.

    Isaac Sim stops an already-playing timeline in its constructor. Isaac Lab
    has already subscribed its STOP handler at that point; on this installation
    that handler renders until PLAY, preventing the constructor from returning
    and the scene from being created. Resets already suppress this handler, but
    construction does not. Preserve all post-construction STOP/reset behavior
    and restore the original methods even when environment creation fails.

    Import lazily: Isaac Sim must be launched before importing isaaclab.sim.
    The optional type allows the lifecycle to be tested without a simulator.
    """
    if simulation_context_type is None:
        from isaaclab.sim import SimulationContext

        simulation_context_type = SimulationContext

    original_init = simulation_context_type.__init__
    original_stop = simulation_context_type._app_control_on_stop_handle_fn

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        previous = getattr(self, "_dexverse_initializing_simulation", False)
        self._dexverse_initializing_simulation = True
        try:
            original_init(self, *args, **kwargs)
        finally:
            self._dexverse_initializing_simulation = previous

    @wraps(original_stop)
    def on_stop(self, event):
        if getattr(self, "_dexverse_initializing_simulation", False):
            logger.warning("Ignoring timeline STOP during simulation construction; scene loading will continue.")
            return
        return original_stop(self, event)

    simulation_context_type.__init__ = initialize
    simulation_context_type._app_control_on_stop_handle_fn = on_stop
    try:
        yield
    finally:
        simulation_context_type.__init__ = original_init
        simulation_context_type._app_control_on_stop_handle_fn = original_stop
