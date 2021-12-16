from datetime import datetime
import time
import sys
from contextlib import contextmanager


def info(*msg, quiet=False):
    now = datetime.now()
    if not quiet:
        print(f"[{now}]", *msg, file=sys.stderr, flush=True)


@contextmanager
def phase_info(*args, **kwargs):
    start_time = time.time()
    info("START:", *args, **kwargs)
    yield None
    end_time = time.time()
    delta_time = end_time - start_time
    delta_time_rounded = round(delta_time)
    info("END:", *args, f"(took {delta_time_rounded} seconds)", **kwargs)
