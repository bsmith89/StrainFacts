from datetime import datetime
import time
import sys
from contextlib import contextmanager
import logging


@contextmanager
def phase_info(name):
    start_time = time.time()
    logging.info(f"START: {name}")
    yield None
    end_time = time.time()
    delta_time = end_time - start_time
    delta_time_rounded = round(delta_time)
    logging.info(f"END: {name} (took {delta_time_rounded} seconds)")
