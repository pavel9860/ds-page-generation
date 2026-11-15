"""Small helpers shared by the workers."""

import signal
from contextlib import contextmanager


class Timeout(BaseException):
    pass


def _raise(*_):
    raise Timeout


@contextmanager
def time_limit(seconds):
    """Raises Timeout in the calling thread after `seconds` (SIGALRM: Python code and interruptible calls only, not native code)."""
    signal.signal(signal.SIGALRM, _raise)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
