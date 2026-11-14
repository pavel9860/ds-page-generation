"""Link text -> the ds_head space (16-d). Learned on the corpus documents with a known source url: the target is the mean head vector of the
document's in-scope pages. A url is a bag of hashed tokens (host, tld, directory names, words of the path, number shapes), mean-pooled by an
EmbeddingBag and a small MLP. It places the 100M urls of the Common Crawl pdf index near the open points of a selection without opening the files."""
import re
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .paths import WORK

BUCKETS = 1 << 20
MODEL = WORK / "urlmodel.pt"
WORD = re.compile(r"[a-z]+|\d+")


def tokens(url, text=""):
    u = url.lower().split("://", 1)[-1]
    host, _, path = u.partition("/")
    host = host[4:] if host.startswith("www.") else host
    parts = host.split(".")
    dirs = [d for d in path.split("?")[0].split("/") if d]
    t = [f"h:{host}", f"tld:{parts[-1]}", f"dom:{'.'.join(parts[-2:])}", f"n:{min(len(dirs), 6)}"]
    t += [f"d{i}:{d}" for i, d in enumerate(dirs[:3])]
    for w in WORD.findall(path):
        t.append(f"#{min(len(w), 6)}" if w.isdigit() else w)
    t += [f"b:{a}_{b}" for a, b in zip(t[4:], t[5:])]
    t += [f"w:{w}" for w in WORD.findall(text.lower()[:600]) if not w.isdigit()][:60]          # the description
    return t


def encode(urls, texts=None):
    """-> (ids int64, offsets int64) of the hashed tokens of the urls and their descriptions."""
    T = [tokens(u, t) for u, t in zip(urls, texts or [""] * len(urls))]
    off = np.concatenate([[0], np.cumsum([len(t) for t in T])[:-1]])
    ids = np.fromiter((zlib.crc32(w.encode()) % BUCKETS for t in T for w in t), np.int64)
    return torch.from_numpy(ids), torch.from_numpy(off.astype(np.int64))


class Net(nn.Module):
    def __init__(self, dim=128, out=16):
        super().__init__()
        self.e = nn.EmbeddingBag(BUCKETS, dim, mode="mean", sparse=True)
        self.f = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, out))

    def forward(self, ids, off):
        return self.f(self.e(ids, off))


def corpus(cc_source_urls, Z, paths, texts):
    """(urls, texts, targets, hosts): one row per corpus document with a url, a description (descr.corpus) and in-scope pages; target = mean head vector of its pages."""
    from .paths import doc_hash
    d = pd.Series([doc_hash(p) for p, _ in paths])
    M = pd.DataFrame(Z).groupby(d.to_numpy()).mean()
    U = cc_source_urls()
    k = [h for h in M.index if U.get(h) and len(texts.get(h, "")) >= 30]
    urls = [U[h] for h in k]
    return urls, [texts[h] for h in k], M.loc[k].to_numpy(np.float32), [u.split("://", 1)[-1].split("/", 1)[0] for u in urls]


def train(urls, texts, Y, hosts, epochs=8, bs=256, r=None):
    """Held-out hosts (10 %) give the error: mean distance of the predicted to the true vector, in units of the global mean and of r."""
    torch.manual_seed(0)
    h = pd.Series(hosts).map(lambda x: zlib.crc32(x.encode()) % 10).to_numpy()
    tr, te = np.flatnonzero(h != 0), np.flatnonzero(h == 0)
    X = [encode([urls[i] for i in s], [texts[i] for i in s]) for s in (tr, te)]
    Yt, Ye = torch.from_numpy(Y[tr]), torch.from_numpy(Y[te])
    net = Net()
    opt = [torch.optim.SparseAdam(list(net.e.parameters()), 3e-3), torch.optim.Adam(net.f.parameters(), 3e-3)]
    best = (9e9, None)
    for ep in range(epochs):
        net.train()
        o = np.random.permutation(len(tr))
        for a in range(0, len(o), bs):
            b = o[a:a + bs]
            ids, off = encode([urls[tr[i]] for i in b], [texts[tr[i]] for i in b])
            loss = ((net(ids, off) - Yt[b]) ** 2).sum(1).mean()
            for o_ in opt:
                o_.zero_grad()
            loss.backward()
            for o_ in opt:
                o_.step()
        net.eval()
        with torch.no_grad():
            err = (net(*X[1]) - Ye).norm(dim=1)
        base = (Ye - Yt.mean(0)).norm(dim=1)
        if err.mean() < best[0]:
            best = (err.mean().item(), {k: v.clone() for k, v in net.state_dict().items()})
        print(f"epoch {ep}: held-out error {err.mean():.3f} (mean vector {base.mean():.3f})" + (f", within r {(err <= r).float().mean():.3f} (mean vector {(base <= r).float().mean():.3f})" if r else ""), flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    torch.save(best[1], MODEL)
    return net


def load():
    net = Net()
    net.load_state_dict(torch.load(MODEL))
    return net.eval()


@torch.no_grad()
def predict(net, urls, texts=None):
    return net(*encode(urls, texts)).numpy()
