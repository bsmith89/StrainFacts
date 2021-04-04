
from datetime import datetime
import sys


def info(*msg):
    now = datetime.now()
    print(f'[{now}]', *msg, file=sys.stderr, flush=True)
