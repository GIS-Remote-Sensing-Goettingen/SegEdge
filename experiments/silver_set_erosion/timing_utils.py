import time
import config as cfg


# Defaults with config overrides
DEBUG_TIMING = getattr(cfg, "DEBUG_TIMING", True)
DEBUG_TIMING_VERBOSE = getattr(cfg, "DEBUG_TIMING_VERBOSE", False)


def time_start():
    if not DEBUG_TIMING:
        return None
    return time.perf_counter()


def time_end(label: str, t0):
    if not DEBUG_TIMING or t0 is None:
        return
    dt = time.perf_counter() - t0
    print(f"[time] {label}: {dt:.3f} s")
