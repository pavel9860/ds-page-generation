"""python -m dspages.ingest <step> [RUN]

  exclude SNAPSHOT_GOT_CSV DIR...   images of the snapshot that are in none of the DIRs (dropped by hand) are never ingested
  register [DIR...]                 new documents (acquired pdf, Lens images, images of loose DIRs) -> ids above the largest
  select                            pages per document (budget, most different first); text-layer language of the pdfs
  shards RUN                        768 x 1024 JPEG shards WORK/npz/RUN
  push RUN [LIMIT] | wait RUN | fetch RUN   Kaggle: dataset, GPU kernel (embedding, hw / text boxes, fill / pic), outputs
  merge RUN                         outputs -> emb_out, hw_det, scan_pct; then the embedding language of the rest; lang_res / src_urls
  drop-candidates DIR               corpus pages of the candidate jpgs deleted by hand from DIR (queue.csv) -> out of emb_out, hw_det, scan_pct
  place                             merged documents -> handwritten/, filled_forms/ or untyped_tmp/ (named <hash>_<lang>.<ext>), plan + manifest follow
  check-lang                        held-out accuracy of the embedding language vote
  status"""
import sys

from . import drop, place, kaggle, lang, merge, pages, registry
from .registry import db


def status():
    c = db()
    for q in ("select kind, state, count(*), sum(picked) from docs group by 1, 2", "select lang_src, count(*) from docs group by 1",
              "select count(*), count(shard) from pages", "select count(*) from excluded"):
        print(q.split("from")[1].strip().split(" group")[0], c.execute(q).fetchall())


def main(argv):
    cmd, args = argv[0], argv[1:]
    if cmd == "exclude":
        print("excluded", registry.exclude_dropped(args[0], args[1:]))
    elif cmd == "register":
        print("registered", registry.register(args))
    elif cmd == "select":
        print("docs, pages", pages.select())
        print("text language", lang.text_langs())
    elif cmd == "shards":
        print(pages.build(args[0]))
    elif cmd == "push":
        kaggle.push(args[0], int(args[1]) if len(args) > 1 else 0)
    elif cmd == "wait":
        sys.exit(0 if kaggle.wait(args[0]) else 1)
    elif cmd == "fetch":
        sys.exit(0 if kaggle.fetch(args[0]) else 1)
    elif cmd == "merge":
        print(merge.merge(args[0]))
        print("embedding language", lang.embed_langs())
        print("lang_res / src_urls rows", merge.finish(args[0]))
    elif cmd == "place":
        print(place.place())
    elif cmd == "drop-candidates":
        print(drop.drop_pages(drop.candidate_pages(args[0])))
    elif cmd == "check-lang":
        lang.check_knn()
    elif cmd == "status":
        status()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
