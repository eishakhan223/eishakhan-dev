"""
SENTIRA — Customer Sentiment Analysis
Positivus-inspired: White · Black · Lime Green
Scrollable pages, complete layout, no cut-off elements
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import threading, re, warnings
from collections import Counter
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ═══════════════════════════════════════════════════
#  DESIGN TOKENS
# ═══════════════════════════════════════════════════
W   = "#FFFFFF"
BK  = "#191A23"
LM  = "#B9FF66"
GY  = "#F3F3F3"
GY2 = "#E0E0E0"
MU  = "#767676"
RD  = "#FF4F5A"
AM  = "#FFA826"
GR  = "#2DB560"   # dark green for positive text on light

F_H2    = ("Arial Black", 13, "bold")
F_H3    = ("Arial Black", 10, "bold")
F_BODY  = ("Arial", 10)
F_LABEL = ("Arial", 9)
F_SMALL = ("Arial", 8)
F_BADGE = ("Arial Black", 8, "bold")
F_MONO  = ("Courier New", 9)

# ═══════════════════════════════════════════════════
#  NLP
# ═══════════════════════════════════════════════════
STOPWORDS = set("""a an the is was are were be been have has had do does did will
would could should may might can i me my we our you your he she it its they them
their this that these those of in on at to for with by from up about into through
during before after above below between out off over under again further then once
here there when where why how all both each few more most other some such no nor not
only own same so than too very just but and or as if""".split())

POS_W = set("""great excellent amazing wonderful fantastic good love best perfect
awesome outstanding superb brilliant incredible quality nice happy satisfied recommend
worth beautiful comfortable easy fast quick reliable durable sturdy impressive
delightful pleased enjoy terrific top""".split())

NEG_W = set("""bad terrible awful horrible worst poor broken defective cheap useless
disappointed disappointing waste return refund stopped failed failure problem issue
complaint ugly slow difficult hard broke cheaply""".split())

CATS = ["Electronics","Home & Kitchen","Beauty","Sports","Books"]

POS_T = [
    "Absolutely love this product. Works perfectly and exceeded every expectation.",
    "Great quality and fast delivery. Very happy with my purchase. Highly recommend.",
    "Outstanding product. Worth every penny. My family absolutely loves it.",
    "Build quality is superb. Works exactly as described. Five stars easily.",
    "Best purchase I made this year. Incredible value and amazing performance.",
    "Does exactly what it promises. Top notch quality. Will definitely buy again.",
    "Fantastic product. Quick delivery, perfect packaging. Very satisfied customer.",
    "Works great, looks premium, feels durable. Very happy with this purchase.",
    "Incredible value for money. Quality is far better than expected at this price.",
    "Durable, reliable and looks beautiful. Highly recommend to everyone.",
]
NEG_T = [
    "Terrible product. Stopped working after just two days. Very disappointed.",
    "Complete waste of money. Cheap quality and broke on first use. Do not buy.",
    "Horrible experience. Arrived damaged and customer service was useless.",
    "Nothing like the description. Returning this immediately. Very unhappy.",
    "Stopped working within a week. Cheaply made and absolutely useless product.",
    "Broke the first time I used it. Total waste of money. Avoid at all costs.",
    "Expected much better quality for this price. Completely disappointed.",
    "Defective product. Does not work at all. Terrible quality control.",
    "Misleading description. Complete disappointment. Would give zero stars.",
    "Cheap and flimsy. Never buying from this brand again. Awful product.",
]
NEU_T = [
    "It is okay. Does the job but nothing particularly special about it.",
    "Average product. Not bad but not great either. Does what it should.",
    "Decent quality for the price. Some minor issues but generally acceptable.",
    "Product is fine. Arrived on time and works as expected. Nothing more.",
    "Works as described. Not impressed but not disappointed. Just okay.",
    "Fair enough. Could be better but meets basic requirements for the price.",
    "Acceptable quality. Neither good nor bad. Serves its purpose adequately.",
    "Does the job. Not exciting but functional. Average in every way.",
]


def clean(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(w for w in t.split() if w not in STOPWORDS and len(w) > 2)


def rule_sentiment(text, rating=None):
    if rating is not None:
        r = float(rating)
        if r >= 4: return "Positive", min(0.96, 0.62+(r-4)*0.17)
        if r <= 2: return "Negative", min(0.96, 0.62+(3-r)*0.17)
        return "Neutral", 0.55
    toks = set(clean(text).split())
    p = len(toks & POS_W); n = len(toks & NEG_W); tot = max(p+n,1)
    if p > n: return "Positive", min(0.95, 0.55+p/tot*0.40)
    if n > p: return "Negative", min(0.95, 0.55+n/tot*0.40)
    return "Neutral", 0.50


def top_kw(texts, n=10):
    toks = []
    for t in texts: toks.extend(clean(t).split())
    return Counter(toks).most_common(n*2)[:n]


def generate_reviews(n=800):
    np.random.seed(42)
    records = []
    for _ in range(n):
        cat = np.random.choice(CATS)
        r   = np.random.random()
        if r < 0.55:
            sent   = "Positive"
            rating = int(np.random.choice([4,5], p=[0.35,0.65]))
            text   = np.random.choice(POS_T)
        elif r < 0.80:
            sent   = "Negative"
            rating = int(np.random.choice([1,2], p=[0.60,0.40]))
            text   = np.random.choice(NEG_T)
        else:
            sent   = "Neutral"
            rating = 3
            text   = np.random.choice(NEU_T)
        records.append({
            "review_text":   text,
            "rating":        rating,
            "category":      cat,
            "sentiment":     sent,
            "helpful_votes": int(np.random.exponential(3)),
        })
    return pd.DataFrame(records)


def run_pipeline(df):
    df = df.copy()
    df["clean"] = df["review_text"].apply(clean)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["sentiment"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["clean"], df["label"], test_size=0.2,
        random_state=42, stratify=df["label"])
    tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1,2), min_df=2)
    Xtr = tfidf.fit_transform(X_tr); Xte = tfidf.transform(X_te)
    mdls = {
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, random_state=42),
        "Naive Bayes":         MultinomialNB(alpha=0.5),
        "Random Forest":       RandomForestClassifier(100, random_state=42, n_jobs=-1),
    }
    results = {}
    for name, m in mdls.items():
        m.fit(Xtr, y_tr); preds = m.predict(Xte)
        results[name] = {
            "acc": round(accuracy_score(y_te, preds)*100, 2),
            "cm":  confusion_matrix(y_te, preds),
        }
    best = max(results, key=lambda n: results[n]["acc"])
    return results, best, tfidf, le


# ═══════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════
def bcard(parent, bg=W, border=BK, bd=2, **kw):
    """Bordered card — thick black border Positivus style."""
    outer = tk.Frame(parent, bg=border)
    inner = tk.Frame(outer, bg=bg, **kw)
    inner.pack(padx=bd, pady=bd, fill="both", expand=True)
    return outer, inner


def card_header(parent, tag, tag_bg, tag_fg, title, card_bg):
    hdr = tk.Frame(parent, bg=card_bg)
    hdr.pack(fill="x", padx=16, pady=(14,8))
    tk.Label(hdr, text=f"  {tag}  ", bg=tag_bg, fg=tag_fg,
             font=F_BADGE, padx=4, pady=4).pack(side="left")
    tk.Label(hdr, text=f"  {title}", bg=card_bg,
             fg=W if card_bg==BK else BK,
             font=F_H2).pack(side="left")
    return hdr


# ═══════════════════════════════════════════════════
#  SCROLLABLE FRAME
# ═══════════════════════════════════════════════════
class ScrollFrame(tk.Frame):
    """A frame with a vertical scrollbar — mouse wheel enabled."""
    def __init__(self, parent, bg=W, **kw):
        super().__init__(parent, bg=bg, **kw)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.vsb    = ttk.Scrollbar(self, orient="vertical",
                                     command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win  = self.canvas.create_window((0,0), window=self.inner,
                                                anchor="nw")
        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Bind mouse wheel everywhere
        for widget in (self.canvas, self.inner, self):
            widget.bind("<MouseWheel>",    self._on_mousewheel)
            widget.bind("<Button-4>",      self._scroll_up)
            widget.bind("<Button-5>",      self._scroll_down)

    def _on_frame_configure(self, _=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self.canvas.itemconfig(self._win, width=e.width)

    def _on_mousewheel(self, e):
        self.canvas.yview_scroll(int(-1*(e.delta/120)), "units")

    def _scroll_up(self, _):   self.canvas.yview_scroll(-1, "units")
    def _scroll_down(self, _): self.canvas.yview_scroll( 1, "units")

    def bind_children(self, widget=None):
        """Recursively bind scroll to all children."""
        w = widget or self.inner
        w.bind("<MouseWheel>", self._on_mousewheel)
        w.bind("<Button-4>",   self._scroll_up)
        w.bind("<Button-5>",   self._scroll_down)
        for child in w.winfo_children():
            self.bind_children(child)


# ═══════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════
class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentira — Amazon Sentiment Analysis")
        self.geometry("1380x920")
        self.minsize(1100, 750)
        self.configure(bg=W)
        self.resizable(True, True)

        self.df=self.results=self.best=self.tfidf=self.le=None
        self.ready = False

        self._styles()
        self._build()
        threading.Thread(target=self._init_data, daemon=True).start()

    def _styles(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure("P.Treeview",
                    background=W, foreground=BK,
                    fieldbackground=W, borderwidth=0,
                    rowheight=34, font=F_BODY)
        s.configure("P.Treeview.Heading",
                    background=LM, foreground=BK,
                    font=F_H3, relief="flat", padding=8)
        s.map("P.Treeview",
              background=[("selected", BK)],
              foreground=[("selected", LM)])
        s.configure("TCombobox",
                    fieldbackground=W, background=W,
                    foreground=BK, selectbackground=W,
                    selectforeground=BK)
        s.map("TCombobox", fieldbackground=[("readonly", W)])

    # ── Master build ──────────────────────────────────────
    def _build(self):
        self._build_header()

        # Control bar (not scrollable — stays fixed)
        self._build_control_bar()

        # Stat cards row (fixed)
        self._build_stat_cards()

        # Tab bar (fixed)
        self._build_tabbar()

        # Page area — each page has its own ScrollFrame
        self.page_area = tk.Frame(self, bg=W)
        self.page_area.pack(fill="both", expand=True)

        self.pages = {}
        for k in ("overview","keywords","models","explore"):
            sf = ScrollFrame(self.page_area, bg=W)
            sf.place(relwidth=1, relheight=1)
            self.pages[k] = sf
            self._make_page(k, sf.inner)

        self._build_footer()
        self._show("overview")

    # ── HEADER ────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=BK, height=60)
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        left = tk.Frame(hdr, bg=BK)
        left.pack(side="left", padx=24, pady=10)
        tk.Label(left, text="  SENTIRA  ", bg=LM, fg=BK,
                 font=("Arial Black",13,"bold"),
                 padx=6, pady=6).pack(side="left")
        tk.Label(left, text="  Amazon Product Reviews",
                 bg=BK, fg=W,
                 font=("Arial", 11)).pack(side="left", pady=2)

        right = tk.Frame(hdr, bg=BK)
        right.pack(side="right", padx=24, pady=10)

        self.run_btn = tk.Button(
            right, text="▶  Analyse Reviews",
            bg=LM, fg=BK,
            font=("Arial Black",10,"bold"),
            relief="flat", cursor="hand2",
            padx=18, pady=8,
            activebackground="#A0E855", activeforeground=BK,
            command=self._run)
        self.run_btn.pack(side="left", padx=(0,10))

        for txt, cmd in [("Predict Review", self._predict_popup),
                          ("Load CSV",       self._load_csv),
                          ("Export",         self._export)]:
            b = tk.Button(right, text=txt, bg=BK, fg=W,
                          font=F_LABEL, relief="flat", cursor="hand2",
                          padx=12, pady=8,
                          highlightbackground="#333333",
                          highlightthickness=1,
                          activebackground="#2A2A35",
                          activeforeground=LM, command=cmd)
            b.pack(side="left", padx=4)

    # ── CONTROL BAR ───────────────────────────────────────
    def _build_control_bar(self):
        outer, inner = bcard(self, bg=GY, border=GY2, bd=1)
        outer.pack(fill="x", padx=24, pady=(14,0))

        row = tk.Frame(inner, bg=GY)
        row.pack(fill="x", padx=20, pady=14)

        # Category
        g = tk.Frame(row, bg=GY); g.pack(side="left", padx=(0,24))
        tk.Label(g, text="CATEGORY", bg=GY, fg=MU, font=F_H3).pack(anchor="w")
        self.cat_var = tk.StringVar(value="All Categories")
        ttk.Combobox(g, textvariable=self.cat_var,
                     values=["All Categories"]+CATS,
                     state="readonly", width=17,
                     font=F_BODY).pack(anchor="w", pady=(4,0))

        tk.Frame(row, bg=GY2, width=1).pack(side="left", fill="y", padx=16)

        # Sentiment pills
        g2 = tk.Frame(row, bg=GY); g2.pack(side="left", padx=(0,24))
        tk.Label(g2, text="SENTIMENT", bg=GY, fg=MU, font=F_H3).pack(anchor="w")
        pr = tk.Frame(g2, bg=GY); pr.pack(anchor="w", pady=(4,0))
        self.sent_var = tk.StringVar(value="All")
        self._pill_btns = {}
        for val, abg, afg in [("All",BK,W),("Positive",LM,BK),
                                ("Negative",RD,W),("Neutral",AM,BK)]:
            b = tk.Button(pr, text=val,
                          bg=abg, fg=afg,
                          font=F_BADGE, relief="flat",
                          cursor="hand2", padx=10, pady=5,
                          activebackground=abg, activeforeground=afg,
                          command=lambda v=val: self._set_sent(v))
            b.pack(side="left", padx=(0,6))
            self._pill_btns[val] = b
        self._refresh_pills()

        tk.Frame(row, bg=GY2, width=1).pack(side="left", fill="y", padx=16)

        # Status
        g3 = tk.Frame(row, bg=GY); g3.pack(side="left", fill="x", expand=True)
        tk.Label(g3, text="STATUS", bg=GY, fg=MU, font=F_H3).pack(anchor="w")
        self.status_var = tk.StringVar(value="Ready — press Analyse Reviews to start")
        tk.Label(g3, textvariable=self.status_var,
                 bg=GY, fg=BK, font=F_BODY,
                 wraplength=360, justify="left").pack(anchor="w", pady=(4,0))

    def _set_sent(self, v):
        self.sent_var.set(v); self._refresh_pills()

    def _refresh_pills(self):
        cur = self.sent_var.get()
        cfg = {"All":(BK,W),"Positive":(LM,BK),"Negative":(RD,W),"Neutral":(AM,BK)}
        for val,(abg,afg) in cfg.items():
            b = self._pill_btns[val]
            if val==cur: b.config(bg=abg, fg=afg)
            else:        b.config(bg=GY2, fg=MU)

    # ── STAT CARDS ────────────────────────────────────────
    def _build_stat_cards(self):
        row = tk.Frame(self, bg=W)
        row.pack(fill="x", padx=24, pady=(12,0))
        self.stat_vars = {}
        specs = [
            ("total",    "Total Reviews",  BK, LM, "Amazon dataset loaded"),
            ("pos_pct",  "Positive",       LM, BK, "sentiment score"),
            ("neg_pct",  "Negative",       RD, W,  "sentiment score"),
            ("best_acc", "Best Accuracy",  BK, W,  "ML model performance"),
        ]
        for i,(key,lbl,bg,fg,sub) in enumerate(specs):
            outer, inner = bcard(row, bg=bg)
            outer.pack(side="left", padx=(0,12) if i<3 else 0,
                       expand=True, fill="x")
            var = tk.StringVar(value="—"); self.stat_vars[key]=var
            tk.Label(inner, textvariable=var, bg=bg, fg=fg,
                     font=("Arial Black",40,"bold"),
                     anchor="w").pack(anchor="w", padx=20, pady=(18,2))
            tk.Label(inner, text=lbl, bg=bg, fg=fg,
                     font=F_H2, anchor="w").pack(anchor="w", padx=20)
            tk.Label(inner, text=sub, bg=bg,
                     fg=fg if bg==BK else MU,
                     font=F_LABEL, anchor="w").pack(anchor="w", padx=20, pady=(0,16))

    # ── TAB BAR ───────────────────────────────────────────
    def _build_tabbar(self):
        bar = tk.Frame(self, bg=W)
        bar.pack(fill="x", padx=24, pady=(16,0))
        tk.Frame(bar, bg=GY2, height=2).pack(fill="x", side="bottom")
        self.tab_btns = {}
        for key,lbl in [("overview","Overview"),("keywords","Keywords"),
                         ("models","ML Models"),("explore","Explore")]:
            b = tk.Button(bar, text=lbl, bg=W, fg=MU,
                          font=F_H3, relief="flat", cursor="hand2",
                          padx=22, pady=12, bd=0,
                          activebackground=W, activeforeground=BK,
                          command=lambda k=key: self._show(k))
            b.pack(side="left")
            self.tab_btns[key] = b

    def _show(self, key):
        for k,b in self.tab_btns.items():
            b.config(fg=BK if k==key else MU,
                     bg=LM if k==key else W)
        for k,sf in self.pages.items():
            if k==key: sf.lift()

    # ── FOOTER ────────────────────────────────────────────
    def _build_footer(self):
        ft = tk.Frame(self, bg=BK, height=36)
        ft.pack(fill="x", side="bottom"); ft.pack_propagate(False)
        self.footer_var = tk.StringVar(value="  Ready")
        tk.Label(ft, textvariable=self.footer_var,
                 bg=BK, fg=MU, font=F_MONO).pack(side="left", padx=20, pady=8)
        tk.Label(ft, text="Sentira v3.0  ·  NLP · ML  ",
                 bg=BK, fg="#333333", font=F_MONO).pack(side="right", padx=20)

    # ═══════════════════════════════════════════════════
    #  PAGE ROUTER
    # ═══════════════════════════════════════════════════
    def _make_page(self, key, parent):
        if   key=="overview": self._page_overview(parent)
        elif key=="keywords": self._page_keywords(parent)
        elif key=="models":   self._page_models(parent)
        elif key=="explore":  self._page_explore(parent)

    # ═══════════════════════════════════════════════════
    #  PAGE: OVERVIEW
    # ═══════════════════════════════════════════════════
    def _page_overview(self, pg):
        pg.configure(bg=W)
        wrap = tk.Frame(pg, bg=W)
        wrap.pack(fill="both", expand=True, padx=24, pady=16)

        # ── Row 1: big chart (left 60%) + donut (right 40%) ──
        row1 = tk.Frame(wrap, bg=W)
        row1.pack(fill="x", pady=(0,14))

        # Left: stacked bar by category
        outer_l, inner_l = bcard(row1, bg=W)
        outer_l.pack(side="left", fill="both", expand=True, padx=(0,12))
        card_header(inner_l, "DISTRIBUTION", LM, BK,
                    "Sentiment by Category", W)
        self.ov_fig = Figure(figsize=(7.5, 4.2), facecolor=W)
        self.ov_canvas = FigureCanvasTkAgg(self.ov_fig, master=inner_l)
        self.ov_canvas.get_tk_widget().pack(fill="both", expand=True,
                                             padx=10, pady=(0,12))
        self._ov_placeholder()

        # Right: donut
        outer_r, inner_r = bcard(row1, bg=W)
        outer_r.pack(side="left", fill="both", ipadx=0)
        outer_r.pack_configure(ipadx=0)
        # fixed width right column
        outer_r.pack(side="left", fill="y")
        inner_r.configure(width=380)
        card_header(inner_r, "SPLIT", BK, LM,
                    "Share of Voice", W)
        self.donut_fig = Figure(figsize=(3.8, 4.2), facecolor=W)
        self.donut_canvas = FigureCanvasTkAgg(self.donut_fig, master=inner_r)
        self.donut_canvas.get_tk_widget().pack(fill="both", expand=True,
                                               padx=10, pady=(0,12))

        # ── Row 2: star ratings FULL width ──
        outer2, inner2 = bcard(wrap, bg=BK)
        outer2.pack(fill="x", pady=(0,14))
        card_header(inner2, "RATINGS", LM, BK,
                    "Star Distribution (1–5 ★)", BK)
        self.rating_fig = Figure(figsize=(13, 3.0), facecolor=BK)
        self.rating_canvas = FigureCanvasTkAgg(self.rating_fig, master=inner2)
        self.rating_canvas.get_tk_widget().pack(fill="both", expand=True,
                                                 padx=10, pady=(0,12))
        self._rating_placeholder()

        # ── Row 3: 3 mini KPI cards ──
        row3 = tk.Frame(wrap, bg=W)
        row3.pack(fill="x", pady=(0,16))

        self.kpi_vars = {}
        kpi_specs = [
            ("pos_n",  "Positive Reviews",  LM, BK),
            ("neg_n",  "Negative Reviews",  BK, W),
            ("avg_r",  "Avg Star Rating",   GY, BK),
        ]
        for key, lbl, bg, fg in kpi_specs:
            outer_k, inner_k = bcard(row3, bg=bg)
            outer_k.pack(side="left", expand=True, fill="x",
                         padx=(0,12) if key!="avg_r" else 0)
            var = tk.StringVar(value="—"); self.kpi_vars[key]=var
            tk.Label(inner_k, textvariable=var, bg=bg, fg=fg,
                     font=("Arial Black",32,"bold")).pack(anchor="w", padx=18, pady=(16,2))
            tk.Label(inner_k, text=lbl, bg=bg, fg=fg if bg!=GY else MU,
                     font=F_H3).pack(anchor="w", padx=18, pady=(0,14))

    def _ov_placeholder(self):
        fig = self.ov_fig; fig.clear()
        ax = fig.add_subplot(111); ax.set_facecolor(GY)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(GY2)
        ax.text(0.5,0.5,"Press  ▶ Analyse Reviews  to begin",
                ha="center",va="center",transform=ax.transAxes,
                color=MU,fontsize=12,fontfamily="Arial",style="italic")
        self.ov_canvas.draw()

    def _rating_placeholder(self):
        fig = self.rating_fig; fig.clear()
        ax = fig.add_subplot(111); ax.set_facecolor(BK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.text(0.5,0.5,"Waiting for analysis …",
                ha="center",va="center",transform=ax.transAxes,
                color="#444",fontsize=11,fontfamily="Arial",style="italic")
        self.rating_canvas.draw()

    def _draw_overview(self, df):
        cmc = {"Positive":LM,"Negative":RD,"Neutral":AM}

        # — Category bar chart —
        fig=self.ov_fig; fig.clear(); fig.patch.set_facecolor(W)
        ax=fig.add_subplot(111); ax.set_facecolor(W)
        for sp in ax.spines.values(): sp.set_color(GY2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        pivot = df.groupby(["category","sentiment"]).size().unstack(fill_value=0)
        cats=pivot.index.tolist(); x=np.arange(len(cats)); w=0.24
        for i,s in enumerate(["Positive","Negative","Neutral"]):
            if s in pivot.columns:
                bars=ax.bar(x+i*w, pivot[s].values, width=w,
                            color=cmc[s], label=s,
                            edgecolor=W, linewidth=1.5)
                for bar in bars:
                    h=bar.get_height()
                    if h>0:
                        ax.text(bar.get_x()+bar.get_width()/2,
                                h+1,str(int(h)),
                                ha="center",fontsize=8,
                                color=BK,fontweight="bold")
        ax.set_xticks(x+w)
        ax.set_xticklabels([c.split()[0] for c in cats],
                            fontsize=11,color=BK,fontweight="bold")
        ax.tick_params(colors=MU,labelsize=9)
        ax.grid(True,axis="y",color=GY2,ls="--",alpha=0.8)
        ax.set_ylabel("Reviews",fontsize=9,color=MU)
        ax.legend(fontsize=9,framealpha=0,labelcolor=BK,loc="upper right")
        fig.tight_layout(pad=1.4)
        self.ov_canvas.draw()

        # — Donut —
        fig2=self.donut_fig; fig2.clear(); fig2.patch.set_facecolor(W)
        ax2=fig2.add_subplot(111)
        cnts=df["sentiment"].value_counts()
        lbls=cnts.index.tolist()
        cols=[cmc.get(l,MU) for l in lbls]
        _,_,autos=ax2.pie(cnts.values,labels=None,autopct="%1.0f%%",
                           colors=cols,startangle=90,pctdistance=0.68,
                           wedgeprops=dict(width=0.52,edgecolor=W,linewidth=4))
        for at in autos:
            at.set_fontsize(12); at.set_fontweight("bold"); at.set_color(BK)
        ax2.legend(lbls,loc="lower center",fontsize=9,framealpha=0,
                   labelcolor=BK,ncol=3,bbox_to_anchor=(0.5,-0.06))
        self.donut_canvas.draw()

        # — Star ratings COMPLETE full-width —
        fig3=self.rating_fig; fig3.clear(); fig3.patch.set_facecolor(BK)
        ax3=fig3.add_subplot(111); ax3.set_facecolor(BK)
        for sp in ax3.spines.values(): sp.set_color("#333")
        ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

        rc=df["rating"].value_counts().sort_index()
        # ensure all 5 stars shown
        all_stars = pd.Series(0, index=[1,2,3,4,5])
        rc = rc.reindex(all_stars.index, fill_value=0)

        bc=[RD if r<=2 else AM if r==3 else LM for r in rc.index]
        bars=ax3.bar(rc.index, rc.values, color=bc,
                     edgecolor=BK, linewidth=3, width=0.55)
        ax3.set_xticks(rc.index)
        ax3.set_xticklabels([f"★ {i}" for i in rc.index],
                             fontsize=12,color=W,fontweight="bold")
        ax3.tick_params(colors="#555",labelsize=9)
        ax3.grid(True,axis="y",color="#2A2A2A",ls="--",alpha=0.6)
        ax3.set_ylabel("Reviews",fontsize=9,color="#555")

        for bar,val in zip(bars,rc.values):
            if val > 0:
                ax3.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+rc.max()*0.03,
                         str(val),ha="center",fontsize=11,
                         color=W,fontweight="bold")
        # legend
        from matplotlib.patches import Patch
        legend_el=[Patch(facecolor=LM,label="4–5 ★  Positive"),
                   Patch(facecolor=AM,label="3 ★  Neutral"),
                   Patch(facecolor=RD,label="1–2 ★  Negative")]
        ax3.legend(handles=legend_el,fontsize=9,framealpha=0,
                   labelcolor=W,loc="upper right")

        fig3.tight_layout(pad=1.2)
        self.rating_canvas.draw()

        # KPI mini cards
        total=len(df); cnts2=df["sentiment"].value_counts()
        self.kpi_vars["pos_n"].set(str(cnts2.get("Positive",0)))
        self.kpi_vars["neg_n"].set(str(cnts2.get("Negative",0)))
        avg_r=df["rating"].mean()
        self.kpi_vars["avg_r"].set(f"{avg_r:.1f} ★")

    # ═══════════════════════════════════════════════════
    #  PAGE: KEYWORDS
    # ═══════════════════════════════════════════════════
    def _page_keywords(self, pg):
        pg.configure(bg=W)
        wrap = tk.Frame(pg, bg=W)
        wrap.pack(fill="both", expand=True, padx=24, pady=16)

        row = tk.Frame(wrap, bg=W); row.pack(fill="both", expand=True)

        self.kw_canvases = []
        self.kw_figs = []

        cfgs = [
            ("Positive","TOP POSITIVE WORDS", LM, BK, W,  BK),
            ("Negative","TOP NEGATIVE WORDS", RD, W,  W,  RD),
            ("Neutral", "TOP NEUTRAL WORDS",  BK, LM, BK, W),
        ]
        for i,(sent,tag,tag_bg,tag_fg,card_bg,txt_col) in enumerate(cfgs):
            outer,inner = bcard(row, bg=card_bg)
            outer.pack(side="left", expand=True, fill="both",
                       padx=(0,12) if i<2 else 0)
            card_header(inner, tag, tag_bg, tag_fg, "Keywords", card_bg)

            fig = Figure(figsize=(3.5, 6.0), facecolor=card_bg)
            canvas = FigureCanvasTkAgg(fig, master=inner)
            canvas.get_tk_widget().pack(fill="both", expand=True,
                                         padx=10, pady=(0,12))
            self.kw_figs.append((fig, tag_bg, card_bg, sent, txt_col))
            self.kw_canvases.append(canvas)

            # placeholder
            ax = fig.add_subplot(111); ax.set_facecolor(card_bg)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color("#333" if card_bg==BK else GY2)
            ax.text(0.5,0.5,"Awaiting analysis…",ha="center",va="center",
                    transform=ax.transAxes,color="#444" if card_bg==BK else MU,
                    fontsize=10,fontfamily="Arial",style="italic")
            canvas.draw()

    def _draw_keywords(self, df):
        for (fig,accent,bg,sent,txt_col),canvas in zip(self.kw_figs,self.kw_canvases):
            fig.clear(); fig.patch.set_facecolor(bg)
            ax=fig.add_subplot(111); ax.set_facecolor(bg)
            for sp in ax.spines.values():
                sp.set_color("#333" if bg==BK else GY2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            subset=df[df["sentiment"]==sent]["review_text"].tolist()
            kws=top_kw(subset,n=10)
            if not kws: canvas.draw(); continue
            words,counts=zip(*kws); y=np.arange(len(words))

            # Glow + solid bars
            ax.barh(y,counts,color=accent,alpha=0.18,height=0.75)
            ax.barh(y,counts,color=accent,alpha=0.90,height=0.42)

            ax.set_yticks(y)
            ax.set_yticklabels(words,fontsize=10,color=txt_col,fontweight="bold")
            ax.tick_params(colors=MU,labelsize=8)
            ax.grid(True,axis="x",color="#333" if bg==BK else GY2,
                    ls="--",alpha=0.5)
            for bar_y,val in zip(y,counts):
                ax.text(val+max(counts)*0.02,bar_y,str(val),
                        va="center",fontsize=9,color=txt_col,fontweight="bold")
            fig.tight_layout(pad=1.5)
            canvas.draw()

    # ═══════════════════════════════════════════════════
    #  PAGE: MODELS
    # ═══════════════════════════════════════════════════
    def _page_models(self, pg):
        pg.configure(bg=W)
        wrap = tk.Frame(pg, bg=W)
        wrap.pack(fill="both", expand=True, padx=24, pady=16)

        # Model score cards row
        row1 = tk.Frame(wrap, bg=W); row1.pack(fill="x", pady=(0,14))
        self.mdl_vars = {}
        for i,(name,short,bg,fg) in enumerate([
            ("Logistic Regression","LR",LM,BK),
            ("Naive Bayes",        "NB",BK,W),
            ("Random Forest",      "RF",BK,LM),
        ]):
            outer,inner=bcard(row1,bg=bg)
            outer.pack(side="left",padx=(0,12) if i<2 else 0,
                       expand=True,fill="x")
            tk.Label(inner,text=short,bg=bg,fg=fg,
                     font=("Arial Black",9)).pack(anchor="w",padx=18,pady=(18,0))
            var=tk.StringVar(value="—"); self.mdl_vars[name]=var
            tk.Label(inner,textvariable=var,bg=bg,fg=fg,
                     font=("Arial Black",38,"bold")).pack(anchor="w",padx=18)
            tk.Label(inner,text=name,bg=bg,
                     fg=fg if bg==BK else MU,
                     font=F_LABEL).pack(anchor="w",padx=18,pady=(0,16))

        # Accuracy + CM
        row2 = tk.Frame(wrap, bg=W); row2.pack(fill="both", expand=True)

        outer_l,inner_l=bcard(row2,bg=W)
        outer_l.pack(side="left",fill="both",expand=True,padx=(0,12))
        card_header(inner_l,"ACCURACY",LM,BK,"Model Comparison",W)
        self.acc_fig=Figure(figsize=(4.5,4.0),facecolor=W)
        acc_canvas=FigureCanvasTkAgg(self.acc_fig,master=inner_l)
        acc_canvas.get_tk_widget().pack(fill="both",expand=True,padx=10,pady=(0,12))
        self.acc_canvas=acc_canvas

        outer_r,inner_r=bcard(row2,bg=BK)
        outer_r.pack(side="left",fill="both",expand=True)
        card_header(inner_r,"MATRIX",LM,BK,"Confusion Matrix",BK)
        self.cm_fig=Figure(figsize=(5.0,4.0),facecolor=BK)
        cm_canvas=FigureCanvasTkAgg(self.cm_fig,master=inner_r)
        cm_canvas.get_tk_widget().pack(fill="both",expand=True,padx=10,pady=(0,12))
        self.cm_canvas=cm_canvas

        # Placeholders
        for fig,bg,msg in [(self.acc_fig,W,"Awaiting analysis…"),
                            (self.cm_fig,BK,"Awaiting analysis…")]:
            ax=fig.add_subplot(111); ax.set_facecolor(bg)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5,0.5,msg,ha="center",va="center",
                    transform=ax.transAxes,
                    color=MU if bg==W else "#444",
                    fontsize=10,fontfamily="Arial",style="italic")
        self.acc_canvas.draw(); self.cm_canvas.draw()

    def _draw_models(self, results):
        for name,var in self.mdl_vars.items():
            if name in results: var.set(f"{results[name]['acc']:.1f}%")

        names=list(results.keys())
        accs=[results[n]["acc"] for n in names]
        shorts=["LR","NB","RF"]
        best_i=accs.index(max(accs))

        # Accuracy
        fig=self.acc_fig; fig.clear(); fig.patch.set_facecolor(W)
        ax=fig.add_subplot(111); ax.set_facecolor(W)
        for sp in ax.spines.values(): sp.set_color(GY2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        bc=[LM if i==best_i else GY2 for i in range(len(names))]
        bars=ax.bar(shorts,accs,color=bc,edgecolor=BK,linewidth=2,width=0.5)
        ax.set_ylim(max(0,min(accs)-15),108)
        ax.tick_params(colors=BK,labelsize=11)
        ax.grid(True,axis="y",color=GY2,ls="--")
        for bar,val in zip(bars,accs):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.5,f"{val:.1f}%",
                    ha="center",fontsize=10,fontweight="bold",color=BK)
        fig.tight_layout(pad=1.5); self.acc_canvas.draw()

        # CM
        best_name=names[best_i]
        cm=results[best_name]["cm"]
        classes=["Neg","Neu","Pos"][:cm.shape[0]]
        fig2=self.cm_fig; fig2.clear(); fig2.patch.set_facecolor(BK)
        ax2=fig2.add_subplot(111); ax2.set_facecolor(BK)
        for sp in ax2.spines.values(): sp.set_color("#333")
        lime_cm=LinearSegmentedColormap.from_list("lm",[BK,"#1A3A00",LM])
        ax2.imshow(cm,cmap=lime_cm,aspect="auto",vmin=0)
        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels(classes,fontsize=12,color=W,fontweight="bold")
        ax2.set_yticks(range(len(classes)))
        ax2.set_yticklabels(classes,fontsize=12,color=W,fontweight="bold")
        ax2.set_xlabel("Predicted",fontsize=9,color=MU)
        ax2.set_ylabel("Actual",fontsize=9,color=MU)
        ax2.set_title(f"Best Model: {best_name}",fontsize=9,color=MU)
        thresh=cm.max()/2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j,i,str(cm[i,j]),ha="center",va="center",
                         fontsize=16,fontweight="bold",
                         color=BK if cm[i,j]>thresh else W)
        fig2.tight_layout(pad=1.5); self.cm_canvas.draw()

    # ═══════════════════════════════════════════════════
    #  PAGE: EXPLORE
    # ═══════════════════════════════════════════════════
    def _page_explore(self, pg):
        pg.configure(bg=W)
        wrap = tk.Frame(pg, bg=W)
        wrap.pack(fill="both", expand=True, padx=24, pady=16)

        # Search bar
        outer_s,inner_s=bcard(wrap,bg=W,border=BK,bd=2)
        outer_s.pack(fill="x",pady=(0,12))
        sf=tk.Frame(inner_s,bg=W); sf.pack(fill="x",padx=16,pady=12)
        tk.Label(sf,text="🔍",bg=W,fg=BK,
                 font=("Arial",14)).pack(side="left",padx=(0,8))
        self.srch=tk.StringVar(); self.srch.trace("w",self._filter_tbl)
        tk.Entry(sf,textvariable=self.srch,
                 bg=W,fg=BK,relief="flat",
                 font=("Arial Black",12),
                 width=38,insertbackground=BK).pack(side="left",ipady=4)
        self.tbl_n=tk.StringVar(value="")
        tk.Label(sf,textvariable=self.tbl_n,
                 bg=W,fg=MU,font=F_LABEL).pack(side="right")

        # Table
        outer_t,inner_t=bcard(wrap,bg=W)
        outer_t.pack(fill="both",expand=True)

        cols=("Review","Category","Rating","Sentiment","Helpful")
        self.tree=ttk.Treeview(inner_t,columns=cols,
                                show="headings",style="P.Treeview",
                                height=18)
        for col,w,a in zip(cols,[420,120,65,100,75],
                           ["w","center","center","center","center"]):
            self.tree.heading(col,text=col.upper())
            self.tree.column(col,width=w,anchor=a)
        self.tree.tag_configure("pos",foreground=GR)
        self.tree.tag_configure("neg",foreground=RD)
        self.tree.tag_configure("neu",foreground=AM)
        vsb=ttk.Scrollbar(inner_t,orient="vertical",command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right",fill="y")
        self.tree.pack(fill="both",expand=True,padx=2,pady=2)

    def _load_tbl(self, df):
        self._tbl_data=df.copy(); self._refresh_tbl(df)

    def _refresh_tbl(self, df):
        self.tree.delete(*self.tree.get_children())
        for _,row in df.head(500).iterrows():
            txt=str(row["review_text"])[:80]+("…" if len(str(row["review_text"]))>80 else "")
            sent=row.get("sentiment","—")
            tag={"Positive":"pos","Negative":"neg","Neutral":"neu"}.get(sent,"")
            self.tree.insert("","end",tags=(tag,),values=(
                txt,row.get("category","—"),
                f"★{row.get('rating','?')}",
                sent,row.get("helpful_votes",0)))
        n=len(df)
        self.tbl_n.set(f"{n:,} reviews"+(" · top 500 shown" if n>500 else ""))

    def _filter_tbl(self,*_):
        if not hasattr(self,"_tbl_data"): return
        q=self.srch.get().lower()
        if not q: self._refresh_tbl(self._tbl_data); return
        f=self._tbl_data[self._tbl_data.apply(
            lambda r: q in str(r.values).lower(),axis=1)]
        self._refresh_tbl(f)

    # ═══════════════════════════════════════════════════
    #  DATA & PIPELINE
    # ═══════════════════════════════════════════════════
    def _init_data(self):
        self.df=generate_reviews(800); self.ready=True
        self._load_tbl(self.df)
        self.stat_vars["total"].set(f"{len(self.df):,}")
        self.status_var.set(f"{len(self.df):,} reviews loaded — press ▶ Analyse Reviews")
        self.footer_var.set(f"  {len(self.df):,} Amazon reviews loaded")

    def _run(self):
        if not self.ready:
            messagebox.showinfo("Wait","Data loading…"); return
        self.run_btn.config(state="disabled",text="⏳  Running…")
        self.status_var.set("Running ML pipeline — please wait…")
        threading.Thread(target=self._run_thread,daemon=True).start()

    def _run_thread(self):
        try:
            df=self.df.copy()
            cat=self.cat_var.get()
            if cat not in ("All","All Categories"):
                df=df[df["category"]==cat]
            snt=self.sent_var.get()
            if snt!="All": df=df[df["sentiment"]==snt]
            if len(df)<30:
                self.after(0,lambda: messagebox.showwarning(
                    "Too few","Not enough data for this filter."))
                self.after(0,self._reset_btn); return

            results,best,tfidf,le=run_pipeline(df)
            self.results=results; self.best=best
            self.tfidf=tfidf; self.le=le

            total=len(df)
            cnts=df["sentiment"].value_counts()
            pos=cnts.get("Positive",0); neg=cnts.get("Negative",0)
            best_acc=results[best]["acc"]

            self.stat_vars["total"].set(f"{total:,}")
            self.stat_vars["pos_pct"].set(f"{pos/total*100:.0f}%")
            self.stat_vars["neg_pct"].set(f"{neg/total*100:.0f}%")
            self.stat_vars["best_acc"].set(f"{best_acc:.1f}%")

            self._draw_overview(df)
            self._draw_keywords(df)
            self._draw_models(results)
            self._load_tbl(df)

            msg=f"Done ✓  Best model: {best}  ·  Accuracy: {best_acc:.1f}%"
            self.status_var.set(msg)
            self.footer_var.set(f"  {msg}")
            self.after(0,self._reset_btn)
        except Exception as e:
            self.after(0,lambda: messagebox.showerror("Error",str(e)))
            self.after(0,self._reset_btn)

    def _reset_btn(self):
        self.run_btn.config(state="normal",text="▶  Analyse Reviews")

    # ═══════════════════════════════════════════════════
    #  PREDICT POPUP
    # ═══════════════════════════════════════════════════
    def _predict_popup(self):
        win=tk.Toplevel(self)
        win.title("Predict Review Sentiment")
        win.geometry("560,460")
        try: win.geometry("560x460")
        except: pass
        win.configure(bg=W); win.resizable(False,False)

        hdr=tk.Frame(win,bg=BK,height=54)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr,text="  PREDICT SENTIMENT",
                 bg=BK,fg=W,font=("Arial Black",13,"bold")).pack(side="left",padx=16,pady=12)
        tk.Label(hdr,text="  Sentira AI  ",
                 bg=LM,fg=BK,font=F_BADGE,padx=6,pady=5).pack(side="right",padx=16,pady=10)

        body=tk.Frame(win,bg=W); body.pack(fill="both",expand=True,padx=24,pady=18)
        tk.Label(body,text="Enter your product review:",
                 bg=W,fg=BK,font=F_H3).pack(anchor="w",pady=(0,8))

        txt_outer=tk.Frame(body,bg=BK)
        txt_outer.pack(fill="x",pady=(0,14))
        txt=tk.Text(txt_outer,height=5,bg=W,fg=BK,relief="flat",
                    font=F_BODY,wrap="word",insertbackground=BK,padx=12,pady=10)
        txt.pack(padx=2,pady=2,fill="x")
        ph="e.g. This product is absolutely amazing, exceeded all my expectations…"
        txt.insert("1.0",ph); txt.config(fg=MU)
        def fi(e):
            if txt.get("1.0","end-1c")==ph: txt.delete("1.0","end"); txt.config(fg=BK)
        def fo(e):
            if not txt.get("1.0","end-1c").strip():
                txt.insert("1.0",ph); txt.config(fg=MU)
        txt.bind("<FocusIn>",fi); txt.bind("<FocusOut>",fo)

        res_f=tk.Frame(body,bg=GY); res_f.pack(fill="x",pady=(0,14))
        ri=tk.Frame(res_f,bg=GY); ri.pack(fill="x",padx=14,pady=12)
        rv=tk.StringVar(value="—"); cv=tk.StringVar(value="Enter a review above")
        rl=tk.Label(ri,textvariable=rv,bg=GY,fg=BK,
                    font=("Arial Black",24,"bold"),anchor="w")
        rl.pack(anchor="w")
        tk.Label(ri,textvariable=cv,bg=GY,fg=MU,font=F_BODY,anchor="w").pack(anchor="w")

        def predict():
            review=txt.get("1.0","end-1c").strip()
            if not review or review==ph: return
            sentiment,conf=rule_sentiment(review)
            icons={"Positive":"↑  Positive","Negative":"↓  Negative","Neutral":"→  Neutral"}
            colors={"Positive":GR,"Negative":RD,"Neutral":AM}
            bgs={"Positive":"#E8F8EE","Negative":"#FFECEC","Neutral":"#FFF5E0"}
            rv.set(icons.get(sentiment,sentiment))
            cv.set(f"Confidence: {conf*100:.0f}%")
            c=colors.get(sentiment,BK); b=bgs.get(sentiment,GY)
            rl.config(fg=c,bg=b); ri.config(bg=b); res_f.config(bg=b)

        tk.Button(body,text="  Analyse Review  ",
                  bg=LM,fg=BK,
                  font=("Arial Black",11,"bold"),
                  relief="flat",cursor="hand2",padx=20,pady=10,
                  activebackground="#A0E855",command=predict).pack(anchor="w")

    # ═══════════════════════════════════════════════════
    #  CSV / EXPORT
    # ═══════════════════════════════════════════════════
    def _load_csv(self):
        path=filedialog.askopenfilename(
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return
        try:
            df=pd.read_csv(path)
            df.columns=df.columns.str.lower().str.strip()
            for alt in ["review","text","comment","body"]:
                if alt in df.columns:
                    df.rename(columns={alt:"review_text"},inplace=True); break
            if "review_text" not in df.columns:
                messagebox.showerror("Error","Need column: review_text"); return
            if "category"      not in df.columns: df["category"]="General"
            if "rating"        not in df.columns: df["rating"]=3
            if "helpful_votes" not in df.columns: df["helpful_votes"]=0
            if "sentiment"     not in df.columns:
                df["sentiment"]=df.apply(
                    lambda r: rule_sentiment(r["review_text"],r.get("rating"))[0],axis=1)
            self.df=df; self.ready=True
            self._load_tbl(df)
            self.stat_vars["total"].set(f"{len(df):,}")
            self.status_var.set(f"{len(df):,} reviews loaded from CSV — press ▶ Analyse")
        except Exception as e:
            messagebox.showerror("Load Error",str(e))

    def _export(self):
        if self.df is None:
            messagebox.showinfo("No data","Run analysis first."); return
        path=filedialog.asksaveasfilename(
            defaultextension=".csv",filetypes=[("CSV","*.csv")])
        if not path: return
        self.df.to_csv(path,index=False)
        messagebox.showinfo("Exported",f"Saved to:\n{path}")


if __name__ == "__main__":
    app = SentimentApp()
    app.mainloop()
