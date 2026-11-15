"""Line break opportunities (UAX #14, ICU's line BreakIterator: dictionary breaks for Thai, Lao, Khmer, Burmese, CJK) through the system libicu."""

import ctypes
import glob
import re
from functools import lru_cache


@lru_cache(maxsize=1)
def _icu():
    """ubrk_open, ubrk_next, ubrk_close of the system libicuuc (its functions carry the ICU major version)."""
    path = max(glob.glob("/usr/lib/*/libicuuc.so.[0-9]*") + glob.glob("/usr/lib/libicuuc.so.[0-9]*"), key=len)
    lib, v = ctypes.CDLL(path), "_" + re.search(r"\.so\.(\d+)", path).group(1)

    def fn(name, res, args):
        f = getattr(lib, name + v)
        f.restype, f.argtypes = res, args
        return f

    return (
        fn(
            "ubrk_open",
            ctypes.c_void_p,
            [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int)],
        ),
        fn("ubrk_next", ctypes.c_int32, [ctypes.c_void_p]),
        fn("ubrk_close", None, [ctypes.c_void_p]),
    )


@lru_cache(maxsize=1 << 16)
def breaks(text, kind=2):
    """Offsets (in characters) where a line may end (kind 2, UBRK_LINE) or a word ends (kind 1, UBRK_WORD), the text's end included."""
    open_, next_, close = _icu()
    b = text.encode("utf-16-le")
    buf, err = ctypes.create_string_buffer(b), ctypes.c_int(0)
    h = open_(kind, b"", buf, len(b) // 2, ctypes.byref(err))  # the default locale
    if err.value > 0:
        raise OSError(f"ubrk_open failed: ICU error {err.value}")
    units, p = [], next_(h)
    while p != -1:
        units.append(p)
        p = next_(h)
    close(h)
    at, u = {}, 0  # UTF-16 units -> characters
    for i, c in enumerate(text):
        at[u] = i
        u += 2 if ord(c) > 0xFFFF else 1
    at[u] = len(text)
    return tuple(at[x] for x in units)
