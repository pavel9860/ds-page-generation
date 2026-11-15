"""Google Translate over HTTP (translate.googleapis.com, one request per text)"""
import time

import requests

from .config import TRANSLATE as T


def google(text, tl, sl="auto"):
    """The translation of `text`, None when Google gives none"""
    for k in range(T.tries):
        r = requests.post(T.url, params=dict(client="gtx", sl=sl, tl=tl, dt="t"), data={"q": text}, timeout=30)
        if r.status_code == 200:
            time.sleep(T.gap)
            return "".join(seg[0] for seg in r.json()[0] if seg[0]) or None
        if r.status_code != 429:
            return None
        time.sleep(T.wait * (k + 1))
    return None


def lines(texts, tl, sl="auto"):
    """The translations of the lines of a page, one per line"""
    out = google("\n".join(texts), tl, sl)
    parts = out.split("\n") if out else []
    return parts if len(parts) == len(texts) else [google(t, tl, sl) for t in texts]
