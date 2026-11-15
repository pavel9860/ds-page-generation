"""Download of the candidates (live url first, else the Common Crawl capture), store, bookkeeping in sqlite (resumable: a url with a row is
never fetched again). Every pdf that opens is kept, with its scan pages counted by the corpus rule (corpus_collecting book_text: images over
SCAN_IMAGE_AREA of the page); which pages enter the set is decided per page at placement. Kept files: STORE/pdf/<sha12>.pdf, one per content
(sha256); a file whose sha256[:12] is a corpus hash is not kept (files edited in the corpus keep their old hash: place.py also drops pages that
duplicate a corpus page)."""

import hashlib
import sqlite3
import sys
import time

import pandas as pd
import pymupdf

from .config import MAX_PAGES  # noqa: F401
from .paths import CC_ROOT, DB, MANIFEST_DB, STORE, WORK

COLS = (
    "url text primary key, point int, status text, sha text, pages int, scan_pages int, file text, t real, error text"
)


def db():
    WORK.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=60)
    c.execute(f"create table if not exists got ({COLS})")
    return c


def corpus_hashes():
    """Content hashes (sha256[:12]) of the corpus documents: a download with one of them is a file the corpus has."""
    with sqlite3.connect(f"file:{MANIFEST_DB}?mode=ro", uri=True) as m:
        return {h for (h,) in m.execute("select doc from plan")} | {
            h for (h,) in m.execute("select hash from src_urls")
        }


def check(data):
    """(status, pages, scan pages, error) of downloaded bytes: ok / no_pdf / pages / broken. Runs in a worker process."""
    if not data or data[:5] != b"%PDF-":
        return "no_pdf", 0, 0, ""
    try:
        sys.path.insert(0, str(CC_ROOT))
        from corpus.book_text import _scan

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.page_count == 0 or doc.page_count > MAX_PAGES:
                return "pages", doc.page_count, 0, ""
            return "ok", doc.page_count, sum(bool(_scan(p)) for p in doc), ""
    except Exception as e:  # pymupdf cannot open the file
        return "broken", 0, 0, f"{type(e).__name__}: {e}"[:300]


def rows(cands, batch=5000):
    """The candidate urls (parquet dir) not fetched yet, hosts in turn (one url of every host, then the next...; like corpus.fill: a host
    serves LIVE_HOST_CONNS requests at a time), streamed BATCH rows at a time."""
    import duckdb

    con = duckdb.connect()
    con.execute("set threads to 4")
    con.execute("set memory_limit = '2GB'")
    con.execute(f"attach '{DB}' as s (type sqlite, read_only)")
    total = con.execute(f"select count(*) from '{cands}' anti join s.got using (url)").fetchone()[0]
    r = con.execute(f"""select url, f, o, l, point from '{cands}' anti join s.got using (url)
                        order by (row_number() over (partition by regexp_extract(url, '^https?://([^/:]+)', 1) order by hash(url)) - 1) // 4, hash(regexp_extract(url, '^https?://([^/:]+)', 1)), hash(url)""")

    def gen():
        while chunk := r.fetchmany(batch):
            yield from (dict(zip(("url", "f", "o", "l", "point"), x)) for x in chunk)

    return total, gen()


def run(cands, workers=6, limit=0, fetchers=256, cc_threads=4):
    """Like corpus.fill: FETCHERS threads take the live sites; a failed site goes to CC_THREADS capture threads at Common Crawl's pace
    (discovery.Pacer); WORKERS processes check the bytes; the main thread stores and writes the db. Every queue is bounded."""
    import queue
    import threading
    from concurrent.futures import ProcessPoolExecutor

    sys.path.insert(0, str(CC_ROOT))
    from corpus import discovery

    c = db()
    shas = {s for (s,) in c.execute("select sha from got where sha is not null")}
    corpus = corpus_hashes()
    total, src = rows(cands)
    total = min(total, limit) if limit else total
    todo, cc_q, got = queue.Queue(2 * fetchers), queue.Queue(500), queue.Queue(4 * workers)
    alive, lock = [fetchers + cc_threads], threading.Lock()

    def feed():
        for k, r in enumerate(src):
            if k >= total:
                break
            todo.put(r)
        for _ in range(fetchers):
            todo.put(None)

    def fetcher():
        while (r := todo.get()) is not None:
            data = discovery.fetch_live(r["url"])
            if data:
                got.put((r, data, ""))
            else:
                cc_q.put(r)
        with lock:
            alive[0] -= 1

    def cc_fetcher():
        pacer = PACER
        while True:
            try:
                r = cc_q.get(timeout=2)
            except queue.Empty:
                if alive[0] <= cc_threads:
                    break
                continue
            pacer.wait()
            try:
                got.put((r, discovery.fetch(r, retries=0), ""))
            except discovery.RateLimited:
                pacer.limited()
                cc_q.put(r)
            except Exception as e:  # the capture could not be read: kept as the reason
                got.put((r, None, f"{type(e).__name__}: {e}"[:300]))
        with lock:
            alive[0] -= 1

    PACER = discovery.Pacer()
    threading.Thread(target=feed, daemon=True).start()
    for fn, n in ((fetcher, fetchers), (cc_fetcher, cc_threads)):
        for _ in range(n):
            threading.Thread(target=fn, daemon=True).start()
    (STORE / "pdf").mkdir(parents=True, exist_ok=True)
    t0, n, ok, inflight = time.time(), 0, 0, []
    with ProcessPoolExecutor(workers) as ex:
        try:
            while alive[0] > 0 or not got.empty() or inflight:
                try:
                    r, data, err = got.get(timeout=1)
                    inflight.append((r, data, err, ex.submit(check, data) if data else None))
                except queue.Empty:
                    pass
                while (
                    inflight
                    and (len(inflight) >= 4 * workers or got.empty())
                    and (inflight[0][3] is None or inflight[0][3].done())
                ):
                    r, data, err, fut = inflight.pop(0)
                    status, pages, scans, e = fut.result() if fut else ("fetch_error" if err else "no_pdf", 0, 0, "")
                    sha = file = None
                    if status == "ok":
                        sha = hashlib.sha256(data).hexdigest()
                        if sha in shas:
                            status = "dup"
                        elif sha[:12] in corpus:
                            status = "known"
                        else:
                            shas.add(sha)
                            file = f"pdf/{sha[:12]}.pdf"
                            (STORE / file).write_bytes(data)
                            ok += 1
                    c.execute(
                        "insert or replace into got (url, point, status, sha, pages, scan_pages, file, t, error) values (?,?,?,?,?,?,?,?,?)",
                        (r["url"], int(r["point"]), status, sha, pages, scans, file, time.time(), err or e),
                    )
                    n += 1
                    if n % 1000 == 0:
                        c.commit()
                        rate = n / (time.time() - t0)
                        print(
                            f"[{int(time.time() - t0)} s] {n}/{total} urls, {ok} kept, {rate * 60:.0f} urls/min, "
                            f"eta {int((total - n) / rate)} s",
                            flush=True,
                        )
        finally:
            c.commit()  # an interrupt keeps the rows written so far
    print(f"done: {n} urls, {ok} kept, {n / (time.time() - t0) * 60:.0f} urls/min", flush=True)


def status():
    c = db()
    S = pd.read_sql("select status, count(*) n, count(distinct point) points from got group by status", c)
    print(S.to_string(index=False))
    E = pd.read_sql(
        "select status, substr(error, 1, 80) error, count(*) n from got where error != '' group by 1, 2 order by n desc limit 10",
        c,
    )
    if len(E):
        print("errors:\n" + E.to_string(index=False))
    print(
        "kept files",
        c.execute("select count(*) from got where file is not null").fetchone()[0],
        "points with a kept file",
        c.execute("select count(distinct point) from got where file is not null").fetchone()[0],
    )
    if c.execute("select count(*) from sqlite_master where name = 'placed'").fetchone()[0]:
        print("points filled", c.execute("select count(*) from placed").fetchone()[0])
