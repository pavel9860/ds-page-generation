import numpy as np
from numba import njit
from numba.typed import List as NumbaList

_glyph_cache: dict[tuple, dict[str, tuple]] = {}


def glyph(font, font_key: tuple, ch: str) -> tuple:
    cache = _glyph_cache.setdefault(font_key, {})
    entry = cache.get(ch)
    if entry is None:
        mask, offset = font.getmask2(ch, mode="L")
        if mask.size[0] and mask.size[1]:
            arr = np.frombuffer(bytes(mask), dtype=np.uint8).reshape(mask.size[1], mask.size[0])
        else:
            arr = np.zeros((0, 0), dtype=np.uint8)
        entry = (arr, offset, font.getlength(ch))
        cache[ch] = entry
    return entry


def advance(font, font_key: tuple, ch: str) -> float:
    return glyph(font, font_key, ch)[2]


@njit(cache=True)
def _blit_glyphs(page, arrs, xs_a, ys_a, xe_a, ye_a, x0, y0):
    for k in range(len(arrs)):
        arr = arrs[k]
        xs, ys, xe, ye, px0, py0 = xs_a[k], ys_a[k], xe_a[k], ye_a[k], x0[k], y0[k]
        for row in range(ys, ye):
            for col in range(xs, xe):
                v = np.uint8(255) - arr[row - py0, col - px0]
                if v < page[row, col]:
                    page[row, col] = v


class PageCanvas:
    def __init__(self, w: int, h: int):
        self.page = np.full((h, w), 255, dtype=np.uint8)
        self.w = w
        self.h = h
        self._arrs: list[np.ndarray] = []
        self._fx: list[float] = []
        self._fy: list[int] = []
        self._fox: list[int] = []
        self._foy: list[int] = []
        self._fgw: list[int] = []
        self._fgh: list[int] = []

    def place_char(self, font, font_key: tuple, ch: str, x: float, y: int) -> float:
        arr, (ox, oy), adv = glyph(font, font_key, ch)
        if arr.size:
            self._arrs.append(arr)
            self._fx.append(x)
            self._fy.append(y)
            self._fox.append(ox)
            self._foy.append(oy)
            self._fgh.append(arr.shape[0])
            self._fgw.append(arr.shape[1])
        return adv

    def flush(self) -> np.ndarray:
        n = len(self._arrs)
        if n == 0:
            return self.page
        fx = np.array(self._fx)
        fy = np.array(self._fy, dtype=np.int64)
        fox = np.array(self._fox, dtype=np.int64)
        foy = np.array(self._foy, dtype=np.int64)
        fgw = np.array(self._fgw, dtype=np.int64)
        fgh = np.array(self._fgh, dtype=np.int64)

        x0 = np.round(fx).astype(np.int64) + fox
        y0 = fy + foy
        xs = np.maximum(0, x0)
        ys = np.maximum(0, y0)
        xe = np.minimum(self.w, x0 + fgw)
        ye = np.minimum(self.h, y0 + fgh)
        keep = np.nonzero((xe > xs) & (ye > ys))[0]
        if keep.size:
            _blit_glyphs(self.page, NumbaList([self._arrs[i] for i in keep]),
                         xs[keep], ys[keep], xe[keep], ye[keep], x0[keep], y0[keep])
        return self.page
