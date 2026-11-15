"""Handwritten pages and filled forms by Google Lens visual search, for the open hw / filled points of a ds_targets run (or, without a
run, the gaps and borders of the two classes in the ds_head space). Google blocks automated browsers, so the browser steps run in the in-app
browser, driven by a Haiku agent (`agent` prints its prompt) or by hand.

  queries RUN  -> LENS/queue.csv, LENS/query/<name>.jpg (768 x 1024): per open point of class hw / filled the nearest in-scope page of the class,
                  one per document (--n N instead: the N pages of both classes ranked by max(gap, border), gap = distance to the nearest selected
                  point, border = share of the other class among the 10 nearest pages)
  seeds DIR..  -> appends the pages of the directories (e.g. generated hw pages, filled forms) as queries seed_<dir>_<file>.jpg to the queue
  serve        -> http://127.0.0.1:8765/next hands out the queries, /save stores the visual matches (LENS/save.jsonl)
  worker       -> images of at least 1440 x 1024 from the matches, boxes counted (yolo26s), stored as LENS/got/<hw|filled|none>/<sha12>.jpg
                  (the document name of the corpus), LENS/got.csv (query, url, file, cls, hw, text); dspages.ingest takes it from there
  agent        -> the prompt of the browser agent"""

import collections
import hashlib
import http.server
import json
import sys
import time
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PORT  # noqa: F401
from .paths import LAYOUTS, STORE, TOOLS

LENS = STORE / "lens"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"}
CLS = ("none", "filled", "hw")


def big(w, h):
    return max(w, h) >= 1440 and min(w, h) >= 1024


def tools():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))


def queries(run=None, n=3000):
    tools()
    import ds_metric as DM
    import ds_targets as T
    import torch

    from ..ingest.render import run_jobs

    R = T.pages()
    Z = torch.from_numpy(np.load(T.OUT / "aux" / "Z.npz")["Z"].astype(np.float32))
    c = DM.hw_class(R.hw, R.text)
    rows = []
    if run:
        X = np.load(run / "out" / "targets.npz")
        cls = pd.read_csv(run / "targets.csv").hw.to_numpy()
        Xt = torch.from_numpy(X["Xt"].astype(np.float32))
        for k in (1, 2):
            ix = np.flatnonzero(c == k)
            d, j = torch.cdist(Xt[np.flatnonzero(cls == CLS[k])], Z[ix]).min(1)
            rows.append(
                R.iloc[ix[j.numpy()]]
                .assign(cls=CLS[k], gap=d.numpy(), border=0.0, score=-d.numpy())
                .sort_values("score", ascending=False)
                .drop_duplicates("h")
            )
    else:
        X = np.load(T.OUT / "n200k" / "out" / "targets.npz")
        P = Z[torch.from_numpy(X["pick"][X["from_page"]])]
        for k in (1, 2):
            ix = np.flatnonzero(c == k)
            gap = torch.cat(
                [torch.cdist(Z[ix[s : s + 4096]], P).min(1).values for s in range(0, len(ix), 4096)]
            ).numpy()
            nb = torch.cat(
                [
                    torch.cdist(Z[ix[s : s + 4096]], Z).topk(11, largest=False).indices[:, 1:]
                    for s in range(0, len(ix), 4096)
                ]
            ).numpy()
            rk = lambda v: pd.Series(v).rank(pct=True).to_numpy()
            D = R.iloc[ix].assign(
                cls=CLS[k], gap=gap, border=(c[nb] != k).mean(1), score=np.maximum(rk(gap), rk((c[nb] != k).mean(1)))
            )
            rows.append(D.sort_values("score", ascending=False).drop_duplicates("h").head(n // 2))
    Q = pd.concat(rows)
    Q["r"] = Q.groupby("cls").cumcount()
    Q = Q.sort_values(["r", "cls"]).reset_index(drop=True)  # hw and filled in turn, best first
    Q["name"] = [f"{i:05d}_{c}_{h}_p{p}.jpg" for i, (c, h, p) in enumerate(zip(Q.cls, Q.h, Q.page))]
    (LENS / "query").mkdir(parents=True, exist_ok=True)
    J = {
        i: j
        for i, _, j in run_jobs([(str(LAYOUTS / p), [(i, int(g))]) for i, (p, g) in enumerate(zip(Q.path, Q.page))])
    }
    for i, nm in enumerate(Q.name):
        if i in J:
            (LENS / "query" / nm).write_bytes(J[i])
    Q[Q.index.isin(J)][["name", "cls", "h", "page", "lang", "hw", "text", "gap", "border", "score"]].to_csv(
        LENS / "queue.csv", index=False
    )
    print(len(J), "query pages ->", LENS / "queue.csv", Q.cls.value_counts().to_dict())


def seeds(dirs):
    """Query pages from images of directories, stretched to 768 x 1024; appended to queue.csv (a query already there is kept)."""
    import cv2

    (LENS / "query").mkdir(parents=True, exist_ok=True)
    q = LENS / "queue.csv"
    Q = pd.read_csv(q) if q.exists() else pd.DataFrame(columns=["name", "cls"])
    rows = []
    for d in map(Path, dirs):
        for f in sorted(d.glob("*.jpg")):
            name = f"seed_{d.name}_{f.stem}.jpg"
            if name in set(Q.name):
                continue
            cv2.imwrite(
                str(LENS / "query" / name),
                cv2.resize(cv2.imread(str(f)), (768, 1024), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            rows.append(
                {
                    "name": name,
                    "cls": d.name,
                    "h": f.stem,
                    "lang": f.stem.split("_")[1] if f.stem.count("_") >= 2 else "en",
                }
            )
    pd.concat([Q, pd.DataFrame(rows)]).to_csv(q, index=False)
    print(len(rows), "seed queries appended ->", q)


GO = """<!doctype html><meta charset=utf-8><body>%s<script>const img=%s;if(img)fetch('/q/'+img).then(r=>r.blob()).then(b=>{const fr=new FileReader();
fr.onload=()=>{location.href='https://www.google.com/?hl=en#'+img+'|'+%s+'|'+fr.result.split(',')[1]};fr.readAsDataURL(b)})</script>"""

STEP = r"""(async()=>{const sl=ms=>new Promise(r=>setTimeout(r,ms)),poll=async(f,to)=>{const t=Date.now();while(Date.now()-t<to){const v=f();if(v)return v;await sl(100)}};
if(!location.href.includes('vsrid')){const h=location.hash.slice(1).split('|'),b64=h.slice(2).join('|');sessionStorage.setItem('lensq',h[0]);sessionStorage.setItem('lensl',h[1]||'en');sessionStorage.removeItem('hlset');
(await poll(()=>document.querySelector('[aria-label="Search by image"]'),8000)).click();
const inp=await poll(()=>[...document.querySelectorAll('input[type=file]')].find(i=>/image\/jpeg/.test(i.accept)),8000);
const bin=Uint8Array.from(atob(b64),c=>c.charCodeAt(0)),dt=new DataTransfer();dt.items.add(new File([bin],'q.jpg',{type:'image/jpeg'}));inp.files=dt.files;setTimeout(()=>inp.dispatchEvent(new Event('change',{bubbles:true})),0);return 'up '+h[0]}
const want=sessionStorage.getItem('lensl')||'en',u=new URL(location.href);if(u.searchParams.get('hl')!==want&&!sessionStorage.getItem('hlset')){sessionStorage.setItem('hlset','1');u.searchParams.set('hl',want);setTimeout(()=>{location.href=u.href},50);return 'hl '+want}
const vm=await poll(()=>[...document.querySelectorAll('div,a,span')].find(x=>x.textContent.trim()==='Visual matches'),10000);if(vm)vm.click();
const get=()=>{const s=new Set();for(const m of document.documentElement.innerHTML.matchAll(/\["(https?:\/\/[^"]+?)",(\d+),(\d+)\]/gi))if(!/gstatic|google\./.test(m[1]))s.add(m[1]+'|'+m[3]+'x'+m[2]);return [...s]};
let n=-1,same=0;const t2=Date.now();while(same<3&&Date.now()-t2<12000){window.scrollTo(0,1e6);await sl(400);const c=get().length;same=c===n?same+1:0;n=c}
const a=get(),g=sessionStorage.getItem('lensq');setTimeout(()=>{location.href='http://127.0.0.1:8765/save?g='+g+'&d='+encodeURIComponent(JSON.stringify(a))},100);return 'got '+a.length})()"""

AGENT = """Drive the in-app browser (mcp__Claude_Browser__*) through the Google Lens queries, until the page text says ALL DONE. Do not read, summarise or explain pages.
1. navigate to http://127.0.0.1:8765/next, wait 4 s: the page forwards to https://www.google.com/?hl=en#<query>|<image>.
2. Once, on that google.com page, run with javascript_tool:  localStorage.s = %s ;
3. Then repeat in browser_batch (24 actions per batch): javascript_tool  new Promise(r=>setTimeout(r,5000)).then(()=>eval(localStorage.s)).catch(e=>String(e))
   Each run is one phase: 'up <query>' (the image is uploaded), then 'got <n>' (the matches are saved and the next query loads). Errors and "page navigated" notes are normal: go on.
4. After 3 failed phases in a row, or when the tab is not on google.com or 127.0.0.1:8765, navigate to http://127.0.0.1:8765/next and go on from 2.
Answer only with the number of queries done (stats: LENS/stats.txt)."""


def serve():
    def queue():  # read again at every request: seeds can be appended while the server runs
        QD = pd.read_csv(LENS / "queue.csv")
        return QD.name.tolist(), (
            dict(zip(QD.name, QD.lang.fillna("en"))) if "lang" in QD else {}
        )  # the browser language of a query: the results prefer pages in it

    out = {}  # query -> time handed out: tabs working in parallel get different queries

    def done():
        f = LENS / "save.jsonl"
        if not f.exists():
            return set()
        n = collections.Counter()  # an empty result (an error page) is tried once more
        for l in open(f):
            r = json.loads(l)
            n[r["g"]] += 1 if r["d"] else 0.5
        return {g for g, v in n.items() if v >= 1}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_OPTIONS(self):
            self.send_response(204)
            for k, v in (
                ("Access-Control-Allow-Origin", "*"),
                ("Access-Control-Allow-Private-Network", "true"),
                ("Access-Control-Allow-Headers", "*"),
            ):
                self.send_header(k, v)
            self.end_headers()

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if u.path == "/s.js":
                body, ct = STEP.encode(), "text/javascript"
            elif u.path.startswith("/q/"):
                body, ct = (LENS / "query" / urllib.parse.unquote(u.path[3:])).read_bytes(), "image/jpeg"
            else:
                if u.path == "/save" and q.get("g", ["null"])[0] not in (
                    "",
                    "null",
                ):  # results of one query (a tab without a query saves nothing), then the next query
                    with open(LENS / "save.jsonl", "a") as f:
                        f.write(json.dumps({"t": time.time(), "g": q["g"][0], "d": json.loads(q["d"][0])}) + "\n")
                Q, lang = queue()
                d = done()
                out.pop(q.get("g", [""])[0], None)
                nxt = next((n for n in Q if n not in d and time.time() - out.get(n, 0) > 120), None)
                out[nxt] = time.time()
                body = (
                    GO
                    % (
                        f"query {nxt} ({len(d)}/{len(Q)} done)" if nxt else "ALL DONE",
                        json.dumps(nxt),
                        json.dumps(lang.get(nxt, "en")),
                    )
                ).encode()
                ct = "text/html"
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

    print(f"http://127.0.0.1:{PORT}/next", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


def fetch(job):
    """(query, url) -> (query, url, jpeg bytes of an image of at least 1440 x 1024, else None)."""
    import cv2
    import httpx

    g, u = job
    try:
        i = cv2.imdecode(
            np.frombuffer(httpx.get(u, headers=UA, timeout=20, follow_redirects=True).content, np.uint8),
            cv2.IMREAD_COLOR,
        )
        if i is None or not big(i.shape[1], i.shape[0]):
            return g, u, None
        return g, u, cv2.imencode(".jpg", i, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    except Exception:
        return g, u, None


def detect(images):
    """jpeg bytes -> (hw, text) box counts of the 768 x 1024 page, as the corpus detection (yolo26s, conf 0.25)."""
    import os

    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):  # torch.set_num_threads alone leaves the OpenMP pool at all cores
        os.environ[k] = "1"
    os.nice(15)
    import cv2
    import torch
    from ultralytics import YOLO

    tools()
    import kaggle_topup as KT

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    m = YOLO(str(KT.WEIGHTS))
    out = []
    for b in images:
        c = (
            m.predict(
                cv2.resize(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), (768, 1024)),
                imgsz=1024,
                conf=0.25,
                verbose=False,
            )[0]
            .boxes.cls.int()
            .numpy()
        )
        out.append((int((c == 0).sum()), int((c == 1).sum())))
    return out


def worker():
    tools()
    import ds_metric as DM

    for c in CLS:
        (LENS / "got" / c).mkdir(parents=True, exist_ok=True)
    log = LENS / "got.csv"
    seen = set(pd.read_csv(log).url) if log.exists() else set()
    pos, _t0 = 0, time.time()
    with ThreadPoolExecutor(32) as io, ProcessPoolExecutor(2) as cpu:
        while True:
            S = [json.loads(l) for l in open(LENS / "save.jsonl")] if (LENS / "save.jsonl").exists() else []
            new, pos = S[pos:], len(S)
            jobs = [
                (r["g"], x.rsplit("|", 1)[0])
                for r in new
                for x in r["d"]
                if big(*map(int, x.rsplit("|", 1)[1].split("x")))
            ]
            jobs = list({u: (g, u) for g, u in jobs if u not in seen}.values())  # one job per url
            seen |= {u for _, u in jobs}
            got = [r for r in io.map(fetch, jobs) if r[2]]
            have = {p.name for c in CLS for p in (LENS / "got" / c).glob("*.jpg")}
            got = list(
                {hashlib.sha256(b).hexdigest()[:12]: (g, u, b) for g, u, b in got}.items()
            )  # one image per content
            got = [(k, g, u, b) for k, (g, u, b) in got if f"{k}.jpg" not in have]
            if got:
                parts = [p for p in np.array_split(np.arange(len(got)), 2) if len(p)]
                box = [x for r in cpu.map(detect, [[got[i][3] for i in p] for p in parts]) for x in r]
                rows = []
                for (k, g, u, b), (h, t) in zip(got, box):
                    c = CLS[int(DM.hw_class([h], [t])[0])]
                    (LENS / "got" / c / f"{k}.jpg").write_bytes(b)
                    rows.append(
                        {"query": g, "url": u, "file": f"{c}/{k}.jpg", "cls": c, "hw": h, "text": t, "t": time.time()}
                    )
                pd.DataFrame(rows).to_csv(log, mode="a", header=not log.exists(), index=False)
            if new or got:
                L = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=["query", "cls"])
                el = (S[-1]["t"] - S[0]["t"]) / 60 if len(S) > 1 else 0
                n = L.cls.value_counts().to_dict()
                (LENS / "stats.txt").write_text(
                    f"queries {len(S)} in {el:.0f} min; downloaded {len(L)}: {n}; per query {len(L) / max(len(S), 1):.1f}, "
                    f"per min hw {n.get('hw', 0) / max(el, 1e-9):.1f} filled {n.get('filled', 0) / max(el, 1e-9):.1f}\n"
                )
            time.sleep(20)


def agent():
    print(AGENT % json.dumps(STEP))
