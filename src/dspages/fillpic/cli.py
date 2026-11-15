"""python -m dspages.fillpic <command>: build the training data, train, check, show, and drive Kaggle."""

import argparse

from . import evaluate, kaggle, show, train
from .config import CONFIG
from .paths import TESTS


def _show(a):
    out = TESTS / a.out
    sources = {
        "folder": lambda: show.run(show.folder_pages(a.folder), "folder", out),
        "hw": lambda: show.run(show.hw_pages(a.skip, a.n), "hw", out),
        "hw-new": lambda: show.run(show.hw_new_pages(a.n, out), "hw", out),
        "random": lambda: show.run(show.random_pages(a.n, a.seed, a.digital, out), "random", out),
    }
    sources[a.source]()


def parser():
    ap = argparse.ArgumentParser(prog="dspages.fillpic")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *args, **opts):
        p = sub.add_parser(name)
        for flag, kw in args:
            p.add_argument(flag, **kw)
        p.set_defaults(f=fn)
        return p

    add("gen", lambda a: train.gen(a.n, a.limit), ("--n", dict(type=int)), ("--limit", dict(type=int, default=0)))
    add("study", lambda a: train.study(a.test), ("--test", dict(action="store_true")))
    add("train", lambda a: train.train())
    add("export", lambda a: kaggle.export())
    add("fill-missing", lambda a: evaluate.fill_missing())
    add("reasons", lambda a: evaluate.no_value_reasons())
    add(
        "recheck",
        lambda a: evaluate.recheck(a.n, a.seed),
        ("--n", dict(type=int, default=3000)),
        ("--seed", dict(type=int, default=1)),
    )
    add("labels", lambda a: show.labels(a.n), ("--n", dict(type=int, default=CONFIG.train.n_pages)))
    add(
        "show",
        _show,
        ("source", dict(choices=["folder", "hw", "hw-new", "random"])),
        ("out", {}),
        ("--folder", {}),
        ("--n", dict(type=int, default=1000)),
        ("--skip", dict(type=int, default=0)),
        ("--seed", dict(type=int, default=0)),
        ("--digital", dict(action="store_true")),
    )
    add("upload-detector", lambda a: kaggle.upload_detector())
    add(
        "push",
        lambda a: kaggle.push(a.kernel, [int(x) for x in a.datasets.split(",")], a.limit, a.only, a.internet),
        ("kernel", {}),
        ("datasets", {}),
        ("--limit", dict(type=int, default=0)),
        ("--only", dict(default="")),
        ("--internet", dict(action="store_true")),
    )
    add(
        "wait",
        lambda a: kaggle.wait(a.kernels.split(","), a.hours),
        ("kernels", {}),
        ("--hours", dict(type=float, default=CONFIG.kaggle.hard_stop_hours)),
    )
    add("fetch", lambda a: kaggle.fetch(a.kernels.split(",")), ("kernels", {}))
    add("rerun", lambda a: kaggle.rerun(a.prefix, a.n), ("prefix", {}), ("n", dict(type=int)))
    add("merge", lambda a: kaggle.merge())
    return ap


def main():
    a = parser().parse_args()
    a.f(a)
