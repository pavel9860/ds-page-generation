import os

os.environ["OMP_NUM_THREADS"]="1"
import json
import random
import sys
from pathlib import Path

from dspages.multilang import filledforms as F
from dspages.multilang import langdoc as L
from dspages.multilang import synthpages as P
from hwsynth.scripts import text_script
from hwsynth.writer import _index

D=Path(sys.argv[1]);need=json.loads((D/"langds"/"forms_need.json").read_text())
index=_index();out=D/"langds"/"forms_blank";out.mkdir(exist_ok=True)
bg=[];skipped=set()
for lang,n in sorted(need.items()):
    pool=P.load_pools(lang,D/"langds",D/"bucketing")
    if len(pool["p"])<3:continue
    for i in range(n):
        rng=random.Random(f"blank{lang}{i}")
        T=P.Text(pool,rng,text_script(" ".join(pool["p"][:6])[:600] or "a"))
        try:font=rng.choice(L.T.fonts_for(" ".join(pool["p"][:4])[:400],index,4))
        except Exception:
            skipped.add(lang);break
        pdf=L.page_pdf(P.body_of("form",T),font,lang,rng,None,fs_max=26)
        if pdf is None:continue
        f=out/f"{lang}_{i}.pdf";f.write_bytes(pdf);bg.append((str(f),0,lang))
print(len(bg),"blank forms written; no font for",sorted(skipped),flush=True)
print(F.prepare(bg,str(D/"langds")),"forms prepared",flush=True)
