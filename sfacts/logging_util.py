from datetime import datetime
import sys


def info(*msg, quiet=False):
    now = datetime.now()
    if not quiet:
        print(f"[{now}]", *msg, file=sys.stderr, flush=True)
