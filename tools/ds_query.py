"""Where do downloads of different queries land in the 3-head page space of the 10K targets (ds_targets, run kept in targets/v10k)?
Groups = (plan type, hw, language) of the targets; per group and query method a few tens of files, the first PAGES pages, rendered like the
shards (form_pages), embedded like the pool (corpus_collecting pagetype/embed_kaggle.py, CPU), through the ds_metric heads.
  host      PDFs of the Common Crawl index from the hosts of the group targets' nearest pages (their source urls)
  keyword   index PDFs whose url holds a word of the group (KEYWORDS) on the group language's TLDs (any TLD for en)
  random    random index PDFs (baseline, one group)
  image     Bing image search in the browser (hw groups): image urls put in OUT/image_urls.json by hand, {group: [url, ...]}
Point set = the 90K existing pages (their vectors) + the 10K target points. Per page: nearest point (target or existing), distance to the
nearest target of its own group, distance to the nearest pool page. Data: ds100k/query (OUT).

python tools/ds_query.py cands [--per 30] | fetch [--per 12] | images | embed | report"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import ds100k as D
import ds_metric as DM
import ds_probe as PR
import ds_targets as T
import form_pages as F
import numpy as np
import pandas as pd

OUT = D.OUT / "query"
V10 = T.OUT / "v10k"
PAGES = 2
GROUPS = [
    ("text", False, "en"),
    ("forms_clean", False, "en"),
    ("flyers", False, "en"),
    ("tables", False, "en"),
    ("bureaucracy", False, "en"),
    ("text", False, "ru"),
    ("forms_clean", False, "ja"),
    ("filled_forms", True, "en"),
    ("bureaucracy", True, "en"),
    ("text", True, "en"),
]
KEYWORDS = {
    "text": ["article", "essay", "story", "text"],
    "forms_clean": ["form", "application", "blank"],
    "flyers": ["flyer", "brochure", "poster"],
    "tables": ["table", "schedule", "price", "list"],
    "bureaucracy": ["decision", "order", "decree", "notice", "minutes"],
    "filled_forms": ["filled", "completed", "signed", "form"],
}
TLD = {"en": None, "ru": ["ru", "su"], "ja": ["jp"]}


def gname(g):
    return f"{g[0]}_{'hw' if g[1] else 'nohw'}_{g[2]}"


def targets():
    G = pd.read_csv(V10 / "targets_10k.csv")
    G["hw"] = G.hw_share >= 0.5
    G["group"] = [gname((t, h, l)) for t, h, l in zip(G.type, G.hw, G.lang)]
    return G


def cmd_cands(a):
    import duckdb

    G = targets()
    urls = PR.source_urls()
    con = duckdb.connect()
    idx = f"'{PR.CC}/cc_pdf_index/*/*.parquet'"
    out = []
    for g in GROUPS:
        t = G[G.group == gname(g)]
        hs = {h for c in ("near1", "near2", "near3") for p in t[c] for h in PR.HEX.findall(p)}
        hosts = sorted({PR.host(urls[h]) for h in hs if h in urls} - {""})
        if hosts:
            con.execute("create or replace temp table h as select unnest(?) as host", [hosts])
            r = con.execute(f"""select url, warc_filename f, warc_record_offset o, warc_record_length l from {idx}
                                where regexp_replace(regexp_extract(url, '^https?://([^/:]+)', 1), '^www\\.', '') in (select host from h)
                                order by hash(url) limit {a.per}""").df()
            out.append(r.assign(group=gname(g), method="host"))
        words = "|".join(KEYWORDS[g[0]])
        tld = "" if TLD[g[2]] is None else f"and regexp_matches(url, '^https?://[^/]+\\.({'|'.join(TLD[g[2]])})(/|:)')"
        r = con.execute(f"""select url, warc_filename f, warc_record_offset o, warc_record_length l from {idx}
                            where regexp_matches(lower(url), '({words})') {tld} order by hash(url) limit {a.per}""").df()
        out.append(r.assign(group=gname(g), method="keyword"))
        print(gname(g), "hosts", len(hosts), flush=True)
    r = con.execute(
        f"select url, warc_filename f, warc_record_offset o, warc_record_length l from {idx} using sample {a.per} rows"
    ).df()
    out.append(r.assign(group="any", method="random"))
    C = pd.concat(out, ignore_index=True).drop_duplicates("url")
    OUT.mkdir(parents=True, exist_ok=True)
    C.to_parquet(OUT / "cands.parquet")
    print(C.groupby(["group", "method"]).size().to_string())


def fetch_pdf(r):
    PR.PAGES = PAGES
    return PR.fetch_one(r)


def cmd_fetch(a):
    sys.path.insert(0, PR.CC)
    C = pd.read_parquet(OUT / "cands.parquet")
    key = list(zip(C.group, C.method))
    got, rows, jpg = {k: 0 for k in key}, [], []
    with ThreadPoolExecutor(6) as ex:
        for (_, r), k, pages in zip(C.iterrows(), key, ex.map(fetch_pdf, [r for _, r in C.iterrows()])):
            if not pages or got[k] >= a.per:
                continue
            got[k] += 1
            for no, j in pages:
                rows.append(dict(group=r.group, method=r.method, url=r.url, page=no, k=len(jpg)))
                jpg.append(j)
    pd.DataFrame(rows).to_parquet(OUT / "pages_pdf.parquet")
    np.savez(OUT / "jpegs_pdf.npz", *[np.frombuffer(j, np.uint8) for j in jpg])
    print("files per group / method", {f"{g}/{m}": n for (g, m), n in got.items()}, "|", len(rows), "pages")


def fetch_image(url):
    import cv2
    import httpx

    try:
        r = httpx.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"},
            timeout=20,
            follow_redirects=True,
        )
        if r.status_code != 200 or len(r.content) > 25_000_000:
            return None
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None or min(img.shape[:2]) < 400:
            return None
        return F.encode(cv2.cvtColor(cv2.resize(img, (F.W, F.H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    except Exception:
        return None


def cmd_images(_):
    U = json.load(open(OUT / "image_urls.json"))
    rows, jpg = [], []
    with ThreadPoolExecutor(6) as ex:
        for g, us in U.items():
            for u, j in zip(us, ex.map(fetch_image, us)):
                if j is not None:
                    rows.append(dict(group=g, method="image", url=u, page=0, k=len(jpg)))
                    jpg.append(j)
    pd.DataFrame(rows).to_parquet(OUT / "pages_img.parquet")
    np.savez(OUT / "jpegs_img.npz", *[np.frombuffer(j, np.uint8) for j in jpg])
    print(pd.DataFrame(rows).groupby("group").size().to_dict(), "images")


def heads():
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    sd = torch.load(DM.OUT / "out" / "head.pt", map_location="cpu")
    f = [nn.Sequential(nn.Linear(1536, 512), nn.GELU(), nn.Linear(512, DM.DIM)) for _ in range(3)]
    for k in range(3):
        f[k].load_state_dict({kk.split(".", 2)[2]: v for kk, v in sd.items() if kk.startswith(f"f.{k}.")})
    msd = json.load(open(DM.OUT / "out" / "metric.json"))["pair_msd"]
    return lambda V: (
        torch.cat([Fn.normalize(f[h](torch.from_numpy(V).float()), dim=1) / msd[h] ** 0.5 for h in range(3)], 1)
        .detach()
        .numpy()
    )


def cmd_embed(_):
    import torch

    sys.path.insert(0, f"{PR.CC}/pagetype")
    import embed_kaggle as K

    torch.set_num_threads(6)
    parts = []
    for s in ("pdf", "img"):
        if (OUT / f"pages_{s}.parquet").exists():
            P = pd.read_parquet(OUT / f"pages_{s}.parquet")
            z = np.load(OUT / f"jpegs_{s}.npz")
            parts.append((P, [torch.from_numpy(z[f"arr_{k}"].copy()) for k in P.k]))
    P = pd.concat([p for p, _ in parts], ignore_index=True)
    raw = [x for _, r in parts for x in r]
    model = K.load_model("cpu")
    V = np.concatenate([K.embed_pages(model, raw[a : a + 4], "cpu")[0] for a in range(0, len(raw), 4)])
    np.save(OUT / "vectors.npy", V.astype(np.float16))
    P.to_parquet(OUT / "pages.parquet")
    print(len(P), "pages embedded")


def cmd_report(_):
    import torch

    P = pd.read_parquet(OUT / "pages.parquet")
    Zq = heads()(np.load(OUT / "vectors.npy").astype(np.float32))
    t = np.load(V10 / "targets.npz")
    Zold = np.load(DM.OUT / "out" / "Z.npz")["Z"].astype(np.float32)
    has = t["dist"] <= json.load(open(V10 / "targets.json"))["r"]
    Ex = Zold[t["page"][has]]
    Xt = t["Xt"].astype(np.float32)
    G = targets()
    pts = torch.from_numpy(np.concatenate([Ex, Xt]))
    q = torch.from_numpy(Zq)
    d = torch.cdist(q, pts)
    j = d.argmin(1).numpy()
    P["nearest"] = np.where(j >= len(Ex), "target", "existing")
    P["d_point"] = d.min(1).values.numpy()
    tgt_group = G.group.to_numpy()[np.clip(j - len(Ex), 0, None)]
    P["target_group"] = np.where(j >= len(Ex), tgt_group, "")
    P["target_type"] = np.where(j >= len(Ex), G.type.to_numpy()[np.clip(j - len(Ex), 0, None)], "")
    dt = torch.cdist(q, torch.from_numpy(Xt))
    P["d_own"] = [
        float(dt[i, torch.from_numpy((G.group == g).to_numpy())].min()) if (G.group == g).any() else np.nan
        for i, g in enumerate(P.group)
    ]
    pool = torch.from_numpy(Zold)
    P["d_pool"] = torch.cat([torch.cdist(q[a : a + 64], pool).min(1).values for a in range(0, len(q), 64)]).numpy()
    P["own_hit"] = P.target_group == P.group
    P.to_parquet(OUT / "result.parquet")
    r = json.load(open(V10 / "targets.json"))
    print(
        f"point spacing: covering radius {r['cover']:.3f}, existing-page radius r {r['r']:.3f}; target point to its nearest pool page median "
        f"{np.median(t['nd'][:, 0]):.3f}"
    )
    S = P.groupby(["group", "method"]).agg(
        pages=("url", "size"),
        files=("url", "nunique"),
        near_target=("nearest", lambda s: (s == "target").mean()),
        own_group_hit=("own_hit", "mean"),
        d_own_target=("d_own", "median"),
        d_point=("d_point", "median"),
        d_pool=("d_pool", "median"),
    )
    print(S.round(3).to_string())
    print("\ntypes of the nearest target (pages that land nearest a target):")
    print(
        P[P.nearest == "target"]
        .groupby(["group", "method"])
        .target_type.agg(lambda s: s.value_counts().head(3).to_dict())
        .to_string()
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("cands").add_argument("--per", type=int, default=30)
    sp.add_parser("fetch").add_argument("--per", type=int, default=12)
    for c in ("images", "embed", "report"):
        sp.add_parser(c)
    a = ap.parse_args()
    {"cands": cmd_cands, "fetch": cmd_fetch, "images": cmd_images, "embed": cmd_embed, "report": cmd_report}[a.cmd](a)
