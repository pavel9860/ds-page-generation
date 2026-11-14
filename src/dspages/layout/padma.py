"""Legacy-font text to Unicode by Padma's own tables and parser (https://github.com/pankaj28843/padma, GPL v2 or later; Telugu, Malayalam, Tamil, Devanagari,
Gujarati, Kannada, Bengali, Gurmukhi: about 100 font names). Padma's JavaScript (encodings, DynamicFontParser, Transformer) runs unchanged in V8
(mini-racer); `convert(font name, text)` is Padma's `Transformer(dynamic fonts -> Unicode)`. The clone is in VENDOR; no Padma code is copied into the repo."""
import os
import re
from functools import lru_cache

from .legacy import CACHE

VENDOR = os.path.join(CACHE, "tables", "padma", "src", "content")
ORDER_FIRST = ["encodings/Common/Padma.js", "encodings/Common/Syllable.js", "encodings/Common/Unicode/Shared.js"]
SKIP = {"defaults.js", "padma.js", "prefs.js", "upgrade.js", "relationships.js", "whitelist.js", "version.js", "contents.rdf"}
SHIM = """var window = this; var Components = {classes: {}, interfaces: {}}; function alert(){}; function dump(){};
var document = {getElementById: function(){return null;}, createElement: function(){return {};}};"""


def _files():
    """The JS files of Padma in the load order of its XUL: Common first, then the encodings of every script, then parsers and transformers."""
    xul = open(os.path.join(VENDOR, "padma.xul")).read() + open(os.path.join(VENDOR, "prefs.xul")).read()
    order = [f for f in re.findall(r'src="([^"]+\.js)"', xul) if os.path.basename(f) not in SKIP]
    seen = []
    for f in ORDER_FIRST + order:
        if f not in seen:
            seen.append(f)
    return seen


@lru_cache(maxsize=1)
def engine():
    from py_mini_racer import MiniRacer
    ctx = MiniRacer()
    ctx.eval(SHIM)
    for f in _files():
        path = os.path.join(VENDOR, f)
        if os.path.exists(path):
            try:
                ctx.eval(open(path, encoding="utf-8", errors="replace").read())      # table definitions: no loop to stop (a timeout only failed them under load)
            except Exception as e:
                raise RuntimeError(f"Padma file {f}: {e}")
    ctx.eval("Unicode.initialize(); Transformer.initialize();")
    ctx.eval("""var __t = Transformer.createTransformer(Transformer.method_DynFonts, Transformer.method_Unicode);
function __conv(face, text) { if (!__t.setDynamicFontByName(face)) return null; return __t.convert(text); }
function __known(face) { return Transformer.isSupportedDynamicFont(face); }""")
    return ctx


def norm(name):
    """A font name as a key: subset tag, case, spaces, hyphens and underscores removed (KOGEBG+Kruti Dev 010 == krutidev010)."""
    return re.sub(r"[^a-z0-9]", "", name.split("+")[-1].lower())


@lru_cache(maxsize=1)
def faces():
    """{normalised name: Padma font face} of the 101 dynamic font encodings."""
    names = engine().eval("Object.keys(Transformer.dynFont_NameToIndex)")
    return {norm(n): engine().eval(f"Transformer.dynFont_Name[Transformer.dynFont_NameToIndex[{n!r}]]") for n in names}


STYLE = re.compile(r"(bold|italic|ital|oblique|regular|normal|norm|plain|light|medium|black|bd|it|b|i)$")


def face_of(name):
    """The Padma font face for a PDF font name (style words at its end dropped when the name itself is not known), else None."""
    k = norm(name)
    f = faces()
    while k:
        if k in f:
            return f[k]
        m = STYLE.search(k)
        if not m or m.start() < 4:
            return None
        k = k[:m.start()]
    return None


def supported(name):
    return face_of(name) is not None


def _face(name):
    return face_of(name) or name


def convert(face, text):
    """Unicode text of `text` typed in the legacy font `face` (None if Padma has no table for the face)."""
    return engine().call("__conv", _face(face), text)
