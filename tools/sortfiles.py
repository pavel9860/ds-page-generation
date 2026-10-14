"""Fast sort-move of jpg/png/pdf files into class folders by one key.

python tools/sortfiles.py
Keys: see KEYS (ctrl+h shows the table), space skips, ctrl+z undoes the last move, ctrl+shift+z redoes it, wheel scrolls pdf pages.
"""
import shutil
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog

import pymupdf
from PIL import Image, ImageTk

from dspages.config import Paths, SortCfg

CFG = SortCfg()
INPUT_DIR = Path(CFG.input_dir)
EXTENSIONS = set(CFG.extensions)
MAX_PDF_PAGES = CFG.max_pdf_pages
PREFETCH = 10
KEYS = {k: (name, Path(Paths().layouts) / folder) for k, name, folder in CFG.keys}


def free_name(path):
    if not path.exists():
        return path
    k = 1
    while (c := path.with_name(f"{path.stem}_{k}{path.suffix}")).exists():
        k += 1
    return c


class App:
    def __init__(self, root):
        self.root = root
        self.files = self.listing(INPUT_DIR) if INPUT_DIR.is_dir() else []
        self.folder = INPUT_DIR
        self.i = 0
        self.done = []
        self.redo = []
        self.photo = None
        self.pool = ThreadPoolExecutor(6)
        self.cache = {}
        self.page_cache = {}
        self.pending = None
        self.canvas = tk.Canvas(root, bg="#222", highlightthickness=0, yscrollincrement=40)
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(root, anchor="w", bg="#111", fg="#ddd")
        self.status.pack(fill="x")
        root.bind("<Key>", self.on_key)
        root.bind("<Control-z>", lambda e: self.undo())
        root.bind("<Control-o>", lambda e: self.open_dir())
        root.bind("<Control-h>", lambda e: self.help())
        root.bind("<Control-Z>", lambda e: self.redo_move())
        root.bind("<Configure>", self.on_resize)
        root.bind("<Button-4>", lambda e: self.scroll(-3))
        root.bind("<Button-5>", lambda e: self.scroll(3))
        root.bind("<MouseWheel>", lambda e: self.scroll(-3 if e.delta > 0 else 3))
        root.after(50, self.show if self.files else self.open_dir)

    def listing(self, folder):
        return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)

    def open_dir(self):
        folder = filedialog.askdirectory(initialdir=self.folder if self.folder.is_dir() else Path(Paths().layouts), parent=self.root)
        if folder:
            self.folder = Path(folder)
            self.files = self.listing(self.folder)
            self.i = 0
            self.done.clear()
            self.redo.clear()
            self.cache.clear()
            self.show()

    def render(self, path, w, h):
        if path.suffix.lower() != ".pdf":
            img = Image.open(path)
            img.draft("RGB", (w, h))
            img = img.convert("RGB")
            scale = min(w / img.width, h / img.height)
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
            return [(img.height, lambda: img)]
        width = w - 4
        with pymupdf.open(path) as doc:
            heights = [round(page.rect.height * width / page.rect.width) for page in list(doc)[:MAX_PDF_PAGES]]
        self.page_future(path, 0, width)
        return [(hgt, (lambda k=k: self.page_future(path, k, width).result())) for k, hgt in enumerate(heights)]

    def render_page(self, path, k, width):
        with pymupdf.open(path) as doc:
            page = doc[k]
            m = width / page.rect.width
            pix = page.get_pixmap(matrix=pymupdf.Matrix(m, m), colorspace=pymupdf.csRGB, alpha=False)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    def page_future(self, path, k, width):
        key = (path, k, width)
        if key not in self.page_cache:
            self.page_cache[key] = self.pool.submit(self.render_page, path, k, width)
        return self.page_cache[key]

    def future(self, path, w, h):
        key = (path, w, h)
        if key not in self.cache:
            self.cache[key] = self.pool.submit(self.render, path, w, h)
        return self.cache[key]

    def on_resize(self, e):
        if e.widget is self.root:
            if self.pending:
                self.root.after_cancel(self.pending)
            self.pending = self.root.after(100, self.show)

    def show(self):
        c = self.canvas
        c.delete("all")
        if self.i >= len(self.files):
            c.create_text(20, 20, anchor="nw", fill="white", font=("monospace", 16), text="No files left. ctrl+z to undo.")
            self.status.config(text=f"done: {len(self.done)} moved")
            return
        path = self.files[self.i]
        w, h = max(c.winfo_width(), 200), max(c.winfo_height(), 200)
        window = self.files[self.i:self.i + 1 + PREFETCH]
        fut = self.future(path, w, h)
        for p in window[1:]:
            self.future(p, w, h)
        keep = {(p, w, h) for p in window + [d[0] for d in self.done[-2:]]}
        for k in [k for k in self.cache if k not in keep]:
            self.cache.pop(k).cancel()
        paths = set(window + [d[0] for d in self.done[-2:]])
        for k in [k for k in self.page_cache if k[0] not in paths]:
            self.page_cache.pop(k).cancel()
        self.shown = {}
        self.pages = []
        try:
            self.pages = fut.result()
        except Exception as ex:
            c.create_text(20, 20, anchor="nw", fill="red", text=f"cannot open: {ex}")
        self.offsets = [sum(q[0] + 6 for q in self.pages[:k]) for k in range(len(self.pages))]
        total = sum(q[0] + 6 for q in self.pages)
        c.config(scrollregion=(0, 0, w, max(total, h)))
        c.yview_moveto(0)
        self.visible()
        self.status.config(text=f"{self.i + 1}/{len(self.files)}  {path.name}  moved: {len(self.done)}  (ctrl+h: help)")

    def visible(self):
        c = self.canvas
        top, h, w = c.canvasy(0), c.winfo_height(), c.winfo_width()
        waiting = False
        for k, (hgt, load) in enumerate(self.pages):
            y = self.offsets[k]
            near = y + hgt > top - h and y < top + 2 * h
            if near and k not in self.shown:
                if self.i < len(self.files) and not self.loaded(k, load):
                    waiting = True
                    continue
                page = load()
                photo = ImageTk.PhotoImage(page)
                self.shown[k] = (c.create_image(max(0, (w - page.width) // 2), y, anchor="nw", image=photo), photo)
            elif not near and k in self.shown:
                c.delete(self.shown.pop(k)[0])
        if waiting:
            self.root.after(20, self.visible)

    def loaded(self, k, load):
        path = self.files[self.i]
        if path.suffix.lower() != ".pdf":
            return True
        width = self.canvas.winfo_width() - 4
        fut = self.page_future(path, k, width)
        return fut.done()

    def scroll(self, n):
        self.canvas.yview_scroll(n, "units")
        self.visible()

    def move(self, key):
        if self.i >= len(self.files):
            return
        src = self.files[self.i]
        if not src.exists():
            del self.files[self.i]
            self.show()
            return
        dst_dir = KEYS[key][1]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = free_name(dst_dir / src.name)
        shutil.move(src, dst)
        self.done.append((src, dst))
        self.redo.clear()
        del self.files[self.i]
        self.show()

    def undo(self):
        if not self.done:
            return
        src, dst = self.done.pop()
        shutil.move(dst, src)
        self.redo.append((src, dst))
        self.files.insert(self.i, src)
        self.show()

    def redo_move(self):
        if not self.redo:
            return
        src, dst = self.redo.pop()
        if self.i < len(self.files) and self.files[self.i] == src:
            del self.files[self.i]
        elif src in self.files:
            self.files.remove(src)
        shutil.move(src, dst)
        self.done.append((src, dst))
        self.show()

    def help(self):
        win = tk.Toplevel(self.root)
        win.title("keys")
        rows = [(k, name, str(d)) for k, (name, d) in KEYS.items()]
        rows += [("space", "skip", ""), ("ctrl+z", "undo last move", ""), ("ctrl+shift+z", "redo", ""), ("wheel", "scroll pdf", ""), ("ctrl+o", "open input dir", ""), ("ctrl+h", "this table", "")]
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                tk.Label(win, text=text, anchor="w", padx=8).grid(row=r, column=c, sticky="w")
        win.bind("<Key>", lambda e: win.destroy())

    def on_key(self, e):
        if e.state & 0x4:
            return
        if e.keysym == "space":
            self.i = min(self.i + 1, len(self.files))
            self.show()
        elif e.keysym in KEYS:
            self.move(e.keysym)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("sortfiles")
    root.geometry("1000x900")
    App(root)
    root.mainloop()
