import time
import logging
import config as cfg


# Defaults with config overrides
DEBUG_TIMING = getattr(cfg, "DEBUG_TIMING", True)
DEBUG_TIMING_VERBOSE = getattr(cfg, "DEBUG_TIMING_VERBOSE", False)

logger = logging.getLogger(__name__)

def time_start():
    """Return a perf counter start if timing is enabled."""
    if not DEBUG_TIMING:
        return None
    return time.perf_counter()


def time_end(label: str, t0):
    """Log elapsed time for a block if timing is enabled."""
    if not DEBUG_TIMING or t0 is None:
        return
    dt = time.perf_counter() - t0
    logger.info("time %s: %.3f s", label, dt)
