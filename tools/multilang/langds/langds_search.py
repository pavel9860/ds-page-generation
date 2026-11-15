import json

from dspages.multilang import pagesearch as S
from dspages.multilang.config import LANGDS as C
from dspages.multilang.inventory import source_files
from dspages.multilang.langds import plan
from dspages.multilang.paths import LANGDS, LAYOUTS

L = str(LAYOUTS)
O = str(LANGDS)
if __name__ == "__main__":
    q, _ = plan([])
    n = source_files(L)
    need = {}
    for (lang, t), k in q.items():
        if t in C.search_folders:
            short = int(k * C.real_share) - sum(n.get((lang, t), [0, 0]))
            if short > 0:
                need.setdefault(lang, {})[t] = short
    json.dump(need, open(O + "/search_needs.json", "w"), indent=0)
    res = {}
    clf = S.classifier()
    for lang in sorted(need):
        res[lang] = S.search(L, lang, need[lang], O + "/search_cache", O + "/found", clf)
        print(lang, "needed", need[lang], "found", res[lang], flush=True)
    json.dump(res, open(O + "/search_found.json", "w"), indent=0)
