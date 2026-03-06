"""
============================================================
  Sales Demand Forecasting System — Interactive GUI
  Run: python app.py
============================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings, os, threading
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ── Colour Palette ────────────────────────────────────────
C = {
    "bg":        "#0F172A",
    "sidebar":   "#1E293B",
    "card":      "#1E293B",
    "border":    "#334155",
    "primary":   "#3B82F6",
    "accent":    "#F59E0B",
    "success":   "#10B981",
    "danger":    "#EF4444",
    "text":      "#F1F5F9",
    "muted":     "#94A3B8",
    "input_bg":  "#0F172A",
    "hover":     "#2563EB",
}

FONT_TITLE  = ("Segoe UI", 18, "bold")
FONT_HEAD   = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 10)

PRODUCTS = [
    "All Products",
    "Detergent_500g", "Shampoo_200ml", "Toothpaste_100g",
    "Body_Lotion_300ml", "Face_Wash_150ml", "Soap_75g",
    "Conditioner_200ml", "Moisturizer_50g"
]
STORES   = ["All Stores", "Store_North", "Store_South", "Store_East", "Store_West"]
MODELS   = ["Linear Regression", "Random Forest", "Gradient Boosting"]

# ══════════════════════════════════════════════════════════
# DATA & ML LOGIC
# ══════════════════════════════════════════════════════════

def generate_dataset():
    np.random.seed(42)
    prods  = PRODUCTS[1:]
    stores = STORES[1:]
    dates  = pd.date_range("2020-01-06", "2023-12-25", freq="W-MON")
    records = []
    for store in stores:
        for product in prods:
            base  = np.random.randint(60, 280)
            slope = np.random.uniform(-0.03, 0.12)
            s_amp = np.random.uniform(0.1, 0.35)
            for i, date in enumerate(dates):
                woy = date.isocalendar()[1]
                m   = date.month
                trend    = base + slope * i
                seasonal = s_amp * base * np.sin(2 * np.pi * woy / 52)
                holiday  = base * (0.25 if m in [12,1] else 0.15 if m in [3,4] else 0)
                promo    = int(np.random.random() < 0.15)
                p_lift   = promo * base * 0.20
                noise    = np.random.normal(0, base * 0.07)
                qty      = max(0, int(trend + seasonal + holiday + p_lift + noise))
                records.append({"date": date, "store": store, "product": product,
                                "sales_qty": qty, "promotion": promo,
                                "price": round(np.random.uniform(50, 500), 2)})
    return pd.DataFrame(records)


def preprocess(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["store","product","date"], inplace=True)
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = (df.groupby(["year_month","store","product"])
               .agg(sales_qty=("sales_qty","sum"), promotion=("promotion","max"),
                    avg_price=("price","mean"))
               .reset_index())
    monthly["date"] = monthly["year_month"].dt.to_timestamp()
    monthly.drop(columns="year_month", inplace=True)
    monthly.sort_values(["store","product","date"], inplace=True)
    monthly.reset_index(drop=True, inplace=True)
    return monthly


def engineer_features(monthly):
    df = monthly.copy()
    grp = df.groupby(["store","product"])["sales_qty"]
    for lag in [1,2,3,6,12]:
        df[f"lag_{lag}m"] = grp.shift(lag)
    df["rolling_3m_mean"]  = grp.shift(1).rolling(3).mean().reset_index(0,drop=True)
    df["rolling_6m_mean"]  = grp.shift(1).rolling(6).mean().reset_index(0,drop=True)
    df["rolling_3m_std"]   = grp.shift(1).rolling(3).std().reset_index(0,drop=True)
    df["rolling_12m_mean"] = grp.shift(1).rolling(12).mean().reset_index(0,drop=True)
    df["month"]      = df["date"].dt.month
    df["quarter"]    = df["date"].dt.quarter
    df["year"]       = df["date"].dt.year
    df["month_sin"]  = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]  = np.cos(2*np.pi*df["month"]/12)
    df["trend_idx"]  = df.groupby(["store","product"]).cumcount()
    le_s = LabelEncoder(); le_p = LabelEncoder()
    df["store_enc"]   = le_s.fit_transform(df["store"])
    df["product_enc"] = le_p.fit_transform(df["product"])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


FEATURE_COLS = [
    "lag_1m","lag_2m","lag_3m","lag_6m","lag_12m",
    "rolling_3m_mean","rolling_6m_mean","rolling_3m_std","rolling_12m_mean",
    "month","quarter","year","month_sin","month_cos",
    "trend_idx","promotion","avg_price","store_enc","product_enc"
]


def train_model(df, model_name):
    cutoff = df["date"].max() - pd.DateOffset(months=6)
    train  = df[df["date"] <= cutoff]
    test   = df[df["date"] >  cutoff]
    X_tr, y_tr = train[FEATURE_COLS], train["sales_qty"]
    X_te, y_te = test[FEATURE_COLS],  test["sales_qty"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=10,
                                                    min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                                        learning_rate=0.05, random_state=42),
    }
    model = models[model_name]
    model.fit(X_tr, y_tr)
    preds = np.maximum(model.predict(X_te), 0)

    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mape = np.mean(np.abs((y_te - preds) / (y_te + 1e-9))) * 100

    return model, train, test, preds, y_te.values, mae, rmse, mape


# ══════════════════════════════════════════════════════════
# GUI APPLICATION
# ══════════════════════════════════════════════════════════

class ForecastApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sales Demand Forecasting System — FMCG")
        self.geometry("1280x820")
        self.minsize(1100, 700)
        self.configure(bg=C["bg"])

        # State
        self.raw_df     = None
        self.monthly_df = None
        self.featured_df = None
        self.model_obj  = None
        self.is_loaded  = False

        self._build_ui()
        self._load_data_thread()   # auto-load on start

    # ── Build UI ─────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg="#1A56DB", height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  📦  Sales Demand Forecasting System",
                 bg="#1A56DB", fg="white", font=("Segoe UI", 14, "bold")).pack(side="left", padx=16, pady=12)
        tk.Label(hdr, text="FMCG · ML-Powered · Python",
                 bg="#1A56DB", fg="#93C5FD", font=FONT_SMALL).pack(side="right", padx=20)

        # Main body
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # Sidebar
        self.sidebar = tk.Frame(body, bg=C["sidebar"], width=260)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # Content area
        self.content = tk.Frame(body, bg=C["bg"])
        self.content.pack(side="left", fill="both", expand=True)
        self._build_content()

        # Status bar
        self.status_var = tk.StringVar(value="  Ready — click Run Forecast to start")
        sb = tk.Frame(self, bg=C["border"], height=28)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        tk.Label(sb, textvariable=self.status_var, bg=C["border"],
                 fg=C["muted"], font=FONT_SMALL).pack(side="left", padx=10, pady=4)

    def _build_sidebar(self):
        pad = {"padx": 16, "pady": 6}

        tk.Label(self.sidebar, text="CONTROLS", bg=C["sidebar"],
                 fg=C["muted"], font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(20,4))

        self._divider()

        # Product selector
        tk.Label(self.sidebar, text="Product", bg=C["sidebar"],
                 fg=C["text"], font=FONT_HEAD).pack(anchor="w", **pad)
        self.product_var = tk.StringVar(value="All Products")
        self.product_cb  = ttk.Combobox(self.sidebar, textvariable=self.product_var,
                                         values=PRODUCTS, state="readonly", width=26)
        self.product_cb.pack(anchor="w", padx=16, pady=(0,8))

        # Store selector
        tk.Label(self.sidebar, text="Store", bg=C["sidebar"],
                 fg=C["text"], font=FONT_HEAD).pack(anchor="w", **pad)
        self.store_var = tk.StringVar(value="All Stores")
        self.store_cb  = ttk.Combobox(self.sidebar, textvariable=self.store_var,
                                       values=STORES, state="readonly", width=26)
        self.store_cb.pack(anchor="w", padx=16, pady=(0,8))

        # Model selector
        tk.Label(self.sidebar, text="Model", bg=C["sidebar"],
                 fg=C["text"], font=FONT_HEAD).pack(anchor="w", **pad)
        self.model_var = tk.StringVar(value="Random Forest")
        self.model_cb  = ttk.Combobox(self.sidebar, textvariable=self.model_var,
                                       values=MODELS, state="readonly", width=26)
        self.model_cb.pack(anchor="w", padx=16, pady=(0,8))

        # Forecast months
        tk.Label(self.sidebar, text="Forecast Horizon (months)", bg=C["sidebar"],
                 fg=C["text"], font=FONT_HEAD).pack(anchor="w", **pad)
        self.horizon_var = tk.IntVar(value=6)
        horizon_frame = tk.Frame(self.sidebar, bg=C["sidebar"])
        horizon_frame.pack(anchor="w", padx=16, pady=(0,8))
        for val in [3, 6, 12]:
            tk.Radiobutton(horizon_frame, text=f"{val}m", variable=self.horizon_var,
                           value=val, bg=C["sidebar"], fg=C["text"],
                           selectcolor=C["primary"], activebackground=C["sidebar"],
                           activeforeground=C["text"], font=FONT_BODY).pack(side="left", padx=4)

        # Show promotions toggle
        self.promo_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.sidebar, text="Include Promotion Effect",
                       variable=self.promo_var, bg=C["sidebar"], fg=C["text"],
                       selectcolor=C["primary"], activebackground=C["sidebar"],
                       activeforeground=C["text"], font=FONT_BODY).pack(anchor="w", padx=16, pady=4)

        self._divider()

        # Run button
        self.run_btn = tk.Button(self.sidebar, text="▶  Run Forecast",
                                  bg=C["primary"], fg="white", font=("Segoe UI", 11, "bold"),
                                  relief="flat", cursor="hand2", pady=10,
                                  command=self._run_forecast)
        self.run_btn.pack(fill="x", padx=16, pady=8)
        self.run_btn.bind("<Enter>", lambda e: self.run_btn.config(bg=C["hover"]))
        self.run_btn.bind("<Leave>", lambda e: self.run_btn.config(bg=C["primary"]))

        # Upload CSV button
        upload_btn = tk.Button(self.sidebar, text="📂  Load Your CSV",
                                bg=C["border"], fg=C["text"], font=FONT_BODY,
                                relief="flat", cursor="hand2", pady=8,
                                command=self._upload_csv)
        upload_btn.pack(fill="x", padx=16, pady=(0,8))

        # Export button
        export_btn = tk.Button(self.sidebar, text="💾  Export Results",
                                bg=C["border"], fg=C["text"], font=FONT_BODY,
                                relief="flat", cursor="hand2", pady=8,
                                command=self._export_results)
        export_btn.pack(fill="x", padx=16, pady=(0,8))

        self._divider()

        # Metrics panel
        tk.Label(self.sidebar, text="LAST RUN METRICS", bg=C["sidebar"],
                 fg=C["muted"], font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(12,4))

        self.metric_frame = tk.Frame(self.sidebar, bg=C["sidebar"])
        self.metric_frame.pack(fill="x", padx=16)

        self.mae_var  = tk.StringVar(value="—")
        self.rmse_var = tk.StringVar(value="—")
        self.mape_var = tk.StringVar(value="—")

        for label, var, color in [
            ("MAE",  self.mae_var,  C["text"]),
            ("RMSE", self.rmse_var, C["text"]),
            ("MAPE", self.mape_var, C["accent"]),
        ]:
            row = tk.Frame(self.metric_frame, bg=C["sidebar"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=C["sidebar"], fg=C["muted"],
                     font=FONT_SMALL, width=6, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, bg=C["sidebar"], fg=color,
                     font=("Consolas", 11, "bold")).pack(side="left")

    def _build_content(self):
        # Tab strip
        tab_frame = tk.Frame(self.content, bg=C["bg"], height=44)
        tab_frame.pack(fill="x")
        tab_frame.pack_propagate(False)

        self.active_tab = tk.StringVar(value="forecast")
        self.tab_btns = {}

        tabs = [
            ("forecast",    "📈  Forecast"),
            ("eda",         "🔍  EDA"),
            ("comparison",  "🤖  Models"),
            ("data",        "📋  Data Table"),
        ]

        for key, label in tabs:
            btn = tk.Button(tab_frame, text=label,
                            bg=C["primary"] if key == "forecast" else C["bg"],
                            fg="white" if key == "forecast" else C["muted"],
                            font=FONT_BODY, relief="flat", padx=18, pady=10,
                            cursor="hand2",
                            command=lambda k=key: self._switch_tab(k))
            btn.pack(side="left")
            self.tab_btns[key] = btn

        # Page container
        self.pages = {}
        container = tk.Frame(self.content, bg=C["bg"])
        container.pack(fill="both", expand=True, padx=12, pady=8)

        for key, _ in tabs:
            page = tk.Frame(container, bg=C["bg"])
            page.place(relwidth=1, relheight=1)
            self.pages[key] = page

        self._build_forecast_page()
        self._build_eda_page()
        self._build_comparison_page()
        self._build_data_page()
        self._switch_tab("forecast")

    def _build_forecast_page(self):
        page = self.pages["forecast"]

        # Top info bar
        info = tk.Frame(page, bg=C["card"], height=60)
        info.pack(fill="x", pady=(0,8))
        info.pack_propagate(False)

        self.info_vars = {}
        for label, key, color in [
            ("Dataset",       "dataset",  C["text"]),
            ("Product",       "product",  C["primary"]),
            ("Model",         "model",    C["accent"]),
            ("Best MAPE",     "mape",     C["success"]),
        ]:
            col = tk.Frame(info, bg=C["card"])
            col.pack(side="left", padx=24, pady=8)
            tk.Label(col, text=label, bg=C["card"], fg=C["muted"],
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            var = tk.StringVar(value="—")
            self.info_vars[key] = var
            tk.Label(col, textvariable=var, bg=C["card"], fg=color,
                     font=("Segoe UI", 12, "bold")).pack(anchor="w")

        # Chart area
        self.forecast_fig = Figure(figsize=(9, 5), facecolor=C["bg"])
        self.forecast_ax  = self.forecast_fig.add_subplot(111)
        self.forecast_ax.set_facecolor("#1E293B")
        self.forecast_ax.text(0.5, 0.5, "Click  ▶ Run Forecast  to generate predictions",
                               ha="center", va="center", transform=self.forecast_ax.transAxes,
                               color=C["muted"], fontsize=13)
        self.forecast_ax.set_xticks([]); self.forecast_ax.set_yticks([])
        self.forecast_canvas = FigureCanvasTkAgg(self.forecast_fig, master=page)
        self.forecast_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_eda_page(self):
        page = self.pages["eda"]
        self.eda_fig = Figure(figsize=(9, 5.5), facecolor=C["bg"])
        self.eda_canvas = FigureCanvasTkAgg(self.eda_fig, master=page)
        self.eda_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_eda_placeholder()

    def _build_comparison_page(self):
        page = self.pages["comparison"]
        self.cmp_fig = Figure(figsize=(9, 5.5), facecolor=C["bg"])
        self.cmp_canvas = FigureCanvasTkAgg(self.cmp_fig, master=page)
        self.cmp_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_data_page(self):
        page = self.pages["data"]

        # Search bar
        search_frame = tk.Frame(page, bg=C["bg"])
        search_frame.pack(fill="x", pady=(0,8))
        tk.Label(search_frame, text="Filter:", bg=C["bg"], fg=C["text"],
                 font=FONT_BODY).pack(side="left", padx=(0,6))
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter_table)
        search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                                bg=C["input_bg"], fg=C["text"], insertbackground=C["text"],
                                relief="flat", font=FONT_MONO, width=30)
        search_entry.pack(side="left", ipady=4, padx=4)

        self.row_count_var = tk.StringVar(value="")
        tk.Label(search_frame, textvariable=self.row_count_var,
                 bg=C["bg"], fg=C["muted"], font=FONT_SMALL).pack(side="left", padx=12)

        # Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                        background=C["card"], foreground=C["text"],
                        fieldbackground=C["card"], borderwidth=0,
                        rowheight=26, font=FONT_SMALL)
        style.configure("Dark.Treeview.Heading",
                        background=C["border"], foreground=C["text"],
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("Dark.Treeview", background=[("selected", C["primary"])])

        cols = ("Date","Store","Product","Sales Qty","Promotion","Avg Price")
        frame = tk.Frame(page, bg=C["bg"])
        frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(frame, columns=cols, show="headings",
                                  style="Dark.Treeview")
        for col in cols:
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._sort_table(c))
            w = 120 if col not in ("Sales Qty","Promotion") else 90
            self.tree.column(col, width=w, anchor="center")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

    # ── Helpers ───────────────────────────────────────────

    def _divider(self):
        tk.Frame(self.sidebar, bg=C["border"], height=1).pack(fill="x", padx=12, pady=8)

    def _switch_tab(self, key):
        self.active_tab.set(key)
        for k, btn in self.tab_btns.items():
            if k == key:
                btn.config(bg=C["primary"], fg="white")
            else:
                btn.config(bg=C["bg"], fg=C["muted"])
        for k, page in self.pages.items():
            if k == key:
                page.lift()

    def _set_status(self, msg, color=None):
        self.status_var.set(f"  {msg}")

    # ── Data Loading ──────────────────────────────────────

    def _load_data_thread(self):
        self._set_status("Loading dataset …")
        t = threading.Thread(target=self._load_data, daemon=True)
        t.start()

    def _load_data(self):
        self.raw_df      = generate_dataset()
        self.monthly_df  = preprocess(self.raw_df)
        self.featured_df = engineer_features(self.monthly_df)
        self.is_loaded   = True
        self._populate_table(self.monthly_df)
        self._draw_eda()
        self._set_status(f"Dataset ready — {len(self.monthly_df):,} monthly records · {len(self.raw_df):,} raw rows")
        self.info_vars["dataset"].set(f"{len(self.monthly_df):,} rows")

    def _upload_csv(self):
        path = filedialog.askopenfilename(
            title="Select Sales CSV",
            filetypes=[("CSV files","*.csv"), ("All files","*.*")]
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            required = {"date","product","sales_qty"}
            if not required.issubset(set(df.columns.str.lower())):
                messagebox.showerror("Column Error",
                    f"CSV must contain: date, product, sales_qty\nFound: {list(df.columns)}")
                return
            df.columns = df.columns.str.lower()
            if "store" not in df.columns:
                df["store"] = "Store_Default"
            if "promotion" not in df.columns:
                df["promotion"] = 0
            if "price" not in df.columns:
                df["price"] = 100.0

            self.raw_df      = df
            self.monthly_df  = preprocess(df)
            self.featured_df = engineer_features(self.monthly_df)
            self.is_loaded   = True
            self._populate_table(self.monthly_df)
            self._draw_eda()
            self._set_status(f"CSV loaded — {len(self.monthly_df):,} monthly records")
            self.info_vars["dataset"].set(f"{len(self.monthly_df):,} rows (CSV)")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # ── Run Forecast ──────────────────────────────────────

    def _run_forecast(self):
        if not self.is_loaded:
            messagebox.showwarning("Not Ready", "Dataset is still loading. Please wait.")
            return

        self.run_btn.config(state="disabled", text="⏳  Running …")
        self._set_status("Training model …")
        t = threading.Thread(target=self._run_forecast_thread, daemon=True)
        t.start()

    def _run_forecast_thread(self):
        try:
            df = self.featured_df.copy()

            # Filter by product
            product = self.product_var.get()
            if product != "All Products":
                df = df[df["product"] == product]

            # Filter by store
            store = self.store_var.get()
            if store != "All Stores":
                df = df[df["store"] == store]

            if len(df) < 20:
                self.after(0, lambda: messagebox.showwarning(
                    "Not Enough Data",
                    "Too few records for this filter combination.\nTry 'All Products' or 'All Stores'."))
                self.after(0, self._reset_run_btn)
                return

            model_name = self.model_var.get()
            model, train, test, preds, y_test, mae, rmse, mape = train_model(df, model_name)

            self.model_obj = model
            self.last_test = test
            self.last_preds = preds
            self.last_y_test = y_test

            # Update metrics
            self.mae_var.set(f"{mae:.1f}")
            self.rmse_var.set(f"{rmse:.1f}")
            self.mape_var.set(f"{mape:.2f}%")

            # Update info bar
            self.info_vars["product"].set(product)
            self.info_vars["model"].set(model_name)
            self.info_vars["mape"].set(f"{mape:.2f}%")

            self._draw_forecast(train, test, preds, y_test, model_name, product, store)
            self._draw_comparison(df, product, store)
            self._set_status(
                f"✓  {model_name} · MAPE={mape:.2f}% · MAE={mae:.1f} · RMSE={rmse:.1f}")
            self.after(0, self._reset_run_btn)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, self._reset_run_btn)

    def _reset_run_btn(self):
        self.run_btn.config(state="normal", text="▶  Run Forecast")

    # ── Drawing ───────────────────────────────────────────

    def _draw_forecast(self, train, test, preds, y_test, model_name, product, store):
        fig  = self.forecast_fig
        fig.clear()
        fig.patch.set_facecolor(C["bg"])

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        for ax in [ax1, ax2]:
            ax.set_facecolor("#1E293B")
            ax.tick_params(colors=C["muted"], labelsize=8)
            ax.spines["bottom"].set_color(C["border"])
            ax.spines["left"].set_color(C["border"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color=C["border"], ls="--", alpha=0.5)

        # Historical + forecast line
        hist = train.groupby("date")["sales_qty"].sum()
        test_agg = test.copy()
        test_agg["predicted"] = preds
        fcast = test_agg.groupby("date").agg(
            actual=("sales_qty","sum"), predicted=("predicted","sum"))

        ax1.plot(hist.index, hist.values, color=C["muted"], lw=1.5, label="Historical")
        ax1.plot(fcast.index, fcast["actual"], color=C["text"], lw=2, marker="o", ms=4, label="Actual")
        ax1.plot(fcast.index, fcast["predicted"], color=C["primary"], lw=2.5,
                 ls="--", marker="s", ms=4, label="Forecast")
        ax1.fill_between(fcast.index, fcast["predicted"]*0.92, fcast["predicted"]*1.08,
                         alpha=0.15, color=C["primary"])
        split = fcast.index.min()
        ax1.axvline(split, color=C["accent"], lw=1.5, ls=":", label="Test Split")
        ax1.set_title(f"Forecast  —  {product}  ·  {store}  ·  {model_name}",
                      color=C["text"], fontsize=11, fontweight="bold")
        ax1.set_ylabel("Units Sold", color=C["muted"], fontsize=9)
        ax1.legend(fontsize=8, facecolor=C["card"], edgecolor=C["border"],
                   labelcolor=C["text"], loc="upper left")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x):,}"))

        # Residuals
        residuals = fcast["actual"] - fcast["predicted"]
        colors_r  = [C["success"] if r >= 0 else C["danger"] for r in residuals]
        ax2.bar(fcast.index, residuals, color=colors_r, width=20, alpha=0.85)
        ax2.axhline(0, color=C["muted"], lw=1)
        ax2.set_title("Residuals  (Actual − Predicted)", color=C["text"], fontsize=10)
        ax2.set_ylabel("Δ Units", color=C["muted"], fontsize=9)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x):,}"))

        fig.tight_layout(pad=2)
        self.forecast_canvas.draw()

    def _draw_eda(self):
        if self.monthly_df is None:
            return
        monthly = self.monthly_df
        fig = self.eda_fig
        fig.clear()
        fig.patch.set_facecolor(C["bg"])

        axes = fig.subplots(2, 2)
        for ax in axes.flat:
            ax.set_facecolor("#1E293B")
            ax.tick_params(colors=C["muted"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(C["border"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color=C["border"], ls="--", alpha=0.4)

        # 1) Trend
        agg = monthly.groupby("date")["sales_qty"].sum()
        roll = agg.rolling(3).mean()
        axes[0,0].fill_between(agg.index, agg.values, alpha=0.15, color=C["primary"])
        axes[0,0].plot(agg.index, agg.values, color=C["primary"], lw=1.5, label="Monthly")
        axes[0,0].plot(roll.index, roll.values, color=C["accent"], lw=2, ls="--", label="3M Avg")
        axes[0,0].set_title("Sales Trend", color=C["text"], fontsize=10, fontweight="bold")
        axes[0,0].legend(fontsize=7, facecolor=C["card"], edgecolor=C["border"], labelcolor=C["text"])
        axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))

        # 2) By product
        by_p = monthly.groupby("product")["sales_qty"].sum().sort_values()
        colors_p = [C["accent"] if v == by_p.max() else C["primary"] for v in by_p.values]
        axes[0,1].barh(by_p.index, by_p.values, color=colors_p, height=0.6)
        axes[0,1].set_title("Sales by Product", color=C["text"], fontsize=10, fontweight="bold")
        axes[0,1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))

        # 3) Seasonality
        monthly["month_n"] = monthly["date"].dt.month
        season = monthly.groupby("month_n")["sales_qty"].mean()
        months = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        bar_c  = [C["danger"] if v==season.max() else C["success"] if v==season.min()
                  else C["primary"] for v in season.values]
        axes[1,0].bar(months, season.values, color=bar_c)
        axes[1,0].set_title("Monthly Seasonality", color=C["text"], fontsize=10, fontweight="bold")
        axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x):,}"))

        # 4) Promo vs no promo
        promo_effect = monthly.groupby("promotion")["sales_qty"].mean()
        labels = ["No Promo", "Promotion"]
        bar_colors = [C["muted"], C["success"]]
        bars = axes[1,1].bar(labels, promo_effect.values, color=bar_colors, width=0.45)
        axes[1,1].set_title("Promotion Effect on Sales", color=C["text"], fontsize=10, fontweight="bold")
        for bar, val in zip(bars, promo_effect.values):
            axes[1,1].text(bar.get_x()+bar.get_width()/2, val*1.01,
                           f"{int(val):,}", ha="center", fontsize=9,
                           color=C["text"], fontweight="bold")
        axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{int(x):,}"))

        fig.tight_layout(pad=2)
        self.eda_canvas.draw()

    def _draw_eda_placeholder(self):
        fig = self.eda_fig
        fig.clear()
        fig.patch.set_facecolor(C["bg"])
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1E293B")
        ax.text(0.5, 0.5, "Loading EDA …", ha="center", va="center",
                transform=ax.transAxes, color=C["muted"], fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        self.eda_canvas.draw()

    def _draw_comparison(self, df, product, store):
        fig = self.cmp_fig
        fig.clear()
        fig.patch.set_facecolor(C["bg"])

        results = {}
        for m_name in MODELS:
            try:
                _, _, _, preds, y_test, mae, rmse, mape = train_model(df, m_name)
                results[m_name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
            except:
                pass

        if not results:
            return

        axes = fig.subplots(1, 3)
        names  = list(results.keys())
        short  = ["LR", "RF", "GB"]
        colors = [C["success"], C["primary"], C["accent"]]

        for ax, metric in zip(axes, ["MAE","RMSE","MAPE"]):
            ax.set_facecolor("#1E293B")
            ax.tick_params(colors=C["muted"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(C["border"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, color=C["border"], ls="--", alpha=0.4, axis="y")

            vals = [results[n][metric] for n in names]
            best = vals.index(min(vals))
            bar_c = [C["accent"] if i==best else c for i,c in enumerate(colors)]
            bars = ax.bar(short, vals, color=bar_c, width=0.5, edgecolor=C["bg"])
            ax.set_title(metric, color=C["text"], fontsize=11, fontweight="bold")
            for bar, val in zip(bars, vals):
                suffix = "%" if metric=="MAPE" else ""
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                        f"{val:.1f}{suffix}", ha="center", va="bottom",
                        fontsize=9, color=C["text"], fontweight="bold")
            ax.set_ylim(0, max(vals)*1.25)

        fig.suptitle(f"Model Comparison  —  {product}  ·  {store}",
                     color=C["text"], fontsize=11, fontweight="bold")
        fig.tight_layout(pad=2)
        self.cmp_canvas.draw()

    # ── Data Table ────────────────────────────────────────

    def _populate_table(self, df):
        self.tree_data = df.copy()
        self._refresh_table(df)

    def _refresh_table(self, df):
        self.tree.delete(*self.tree.get_children())
        for _, row in df.head(500).iterrows():
            self.tree.insert("", "end", values=(
                row["date"].strftime("%Y-%m") if hasattr(row["date"],"strftime") else str(row["date"]),
                row.get("store","—"),
                row["product"],
                f"{int(row['sales_qty']):,}",
                "✓" if row.get("promotion",0) else "—",
                f"{row.get('avg_price',0):.0f}",
            ))
        count = len(df)
        self.row_count_var.set(f"{count:,} rows" + (" (showing first 500)" if count > 500 else ""))

    def _filter_table(self, *_):
        if not hasattr(self, "tree_data"):
            return
        q = self.search_var.get().lower()
        if not q:
            self._refresh_table(self.tree_data)
            return
        filtered = self.tree_data[
            self.tree_data.apply(lambda r: q in str(r.values).lower(), axis=1)
        ]
        self._refresh_table(filtered)

    def _sort_table(self, col):
        if not hasattr(self, "tree_data"):
            return
        col_map = {
            "Date":"date","Store":"store","Product":"product",
            "Sales Qty":"sales_qty","Promotion":"promotion","Avg Price":"avg_price"
        }
        c = col_map.get(col, "date")
        if c in self.tree_data.columns:
            self.tree_data.sort_values(c, ascending=True, inplace=True)
            self._refresh_table(self.tree_data)

    # ── Export ────────────────────────────────────────────

    def _export_results(self):
        if not hasattr(self, "last_preds"):
            messagebox.showinfo("No Results", "Run a forecast first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Results", defaultextension=".csv",
            filetypes=[("CSV","*.csv")])
        if not path:
            return
        test = self.last_test.copy()
        test["predicted"] = self.last_preds
        test.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Results exported to:\n{path}")


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = ForecastApp()
    app.mainloop()
