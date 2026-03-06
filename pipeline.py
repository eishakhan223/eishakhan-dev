"""
============================================================
  Sales Demand Forecasting System for FMCG Products
  Pipeline: Preprocessing → EDA → Features → Models → Eval
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings, os, json
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

PALETTE = {
    "primary":   "#1A56DB",
    "accent":    "#E3A008",
    "danger":    "#E02424",
    "success":   "#057A55",
    "neutral":   "#374151",
    "bg":        "#F9FAFB",
    "grid":      "#E5E7EB",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    "white",
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["neutral"],
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   PALETTE["neutral"],
    "xtick.color":       PALETTE["neutral"],
    "ytick.color":       PALETTE["neutral"],
    "grid.color":        PALETTE["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "font.family":       "DejaVu Sans",
    "legend.framealpha": 0.9,
})

# ══════════════════════════════════════════════════════════
# 1. GENERATE SYNTHETIC DATASET
# ══════════════════════════════════════════════════════════

def generate_dataset():
    np.random.seed(42)
    PRODUCTS = [
        "Detergent_500g", "Shampoo_200ml", "Toothpaste_100g",
        "Body_Lotion_300ml", "Face_Wash_150ml", "Soap_75g",
        "Conditioner_200ml", "Moisturizer_50g"
    ]
    STORES = ["Store_North", "Store_South", "Store_East", "Store_West"]
    date_range = pd.date_range("2020-01-06", "2023-12-25", freq="W-MON")

    records = []
    for store in STORES:
        for product in PRODUCTS:
            base      = np.random.randint(60, 280)
            slope     = np.random.uniform(-0.03, 0.12)
            s_amp     = np.random.uniform(0.1, 0.35)
            for i, date in enumerate(date_range):
                woy = date.isocalendar()[1]
                m   = date.month
                trend    = base + slope * i
                seasonal = s_amp * base * np.sin(2 * np.pi * woy / 52)
                holiday  = base * (0.25 if m in [12,1] else 0.15 if m in [3,4] else 0)
                promo    = int(np.random.random() < 0.15)
                p_lift   = promo * base * 0.20
                noise    = np.random.normal(0, base * 0.07)
                qty      = max(0, int(trend + seasonal + holiday + p_lift + noise))
                records.append({
                    "date": date, "store": store, "product": product,
                    "sales_qty": qty, "promotion": promo,
                    "price": round(np.random.uniform(50, 500), 2)
                })

    df = pd.DataFrame(records)
    print(f"  Rows: {len(df):,}  |  Date range: {df.date.min().date()} → {df.date.max().date()}")
    return df


# ══════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════

def preprocess(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["store","product","date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Missing values
    missing_before = df.isnull().sum().sum()
    df["sales_qty"].fillna(df["sales_qty"].median(), inplace=True)
    df["promotion"].fillna(0, inplace=True)

    # Monthly aggregation (store + product level)
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby(["year_month","store","product"])
        .agg(
            sales_qty  = ("sales_qty", "sum"),
            promotion  = ("promotion", "max"),
            avg_price  = ("price",     "mean"),
        )
        .reset_index()
    )
    monthly["date"] = monthly["year_month"].dt.to_timestamp()
    monthly.drop(columns="year_month", inplace=True)
    monthly.sort_values(["store","product","date"], inplace=True)
    monthly.reset_index(drop=True, inplace=True)

    print(f"  Missing values fixed: {missing_before}  |  Monthly rows: {len(monthly):,}")
    return monthly


# ══════════════════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════

def run_eda(monthly):
    # ── Plot 1: Overall monthly sales trend ──────────────
    agg = monthly.groupby("date")["sales_qty"].sum().reset_index()
    agg["rolling_3m"] = agg["sales_qty"].rolling(3).mean()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Exploratory Data Analysis — FMCG Sales", fontsize=16,
                 fontweight="bold", color=PALETTE["neutral"], y=1.01)

    # Total sales trend
    ax = axes[0,0]
    ax.fill_between(agg["date"], agg["sales_qty"], alpha=0.15, color=PALETTE["primary"])
    ax.plot(agg["date"], agg["sales_qty"], color=PALETTE["primary"], lw=1.5, label="Monthly Sales")
    ax.plot(agg["date"], agg["rolling_3m"], color=PALETTE["accent"], lw=2.5,
            ls="--", label="3-Month Rolling Avg")
    ax.set_title("Total Monthly Sales Trend")
    ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.legend(); ax.grid(True)

    # Sales by product
    ax = axes[0,1]
    by_prod = monthly.groupby("product")["sales_qty"].sum().sort_values(ascending=True)
    colors = [PALETTE["primary"] if i < len(by_prod)-3 else PALETTE["accent"]
              for i in range(len(by_prod))]
    bars = ax.barh(by_prod.index, by_prod.values, color=colors, edgecolor="white", height=0.6)
    ax.set_title("Total Sales by Product")
    ax.set_xlabel("Total Units Sold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
    for bar, val in zip(bars, by_prod.values):
        ax.text(val + by_prod.max()*0.01, bar.get_y()+bar.get_height()/2,
                f"{int(val):,}", va="center", fontsize=8, color=PALETTE["neutral"])
    ax.grid(True, axis="x")

    # Monthly seasonality (avg by month)
    ax = axes[1,0]
    monthly["month_num"] = monthly["date"].dt.month
    season = monthly.groupby("month_num")["sales_qty"].mean()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    bar_colors = [PALETTE["danger"] if v == season.max()
                  else PALETTE["success"] if v == season.min()
                  else PALETTE["primary"] for v in season.values]
    ax.bar(month_labels, season.values, color=bar_colors, edgecolor="white")
    ax.set_title("Average Monthly Seasonality")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Units Sold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.grid(True, axis="y")

    # Sales by store
    ax = axes[1,1]
    by_store = monthly.groupby("store")["sales_qty"].sum().sort_values(ascending=False)
    total = by_store.sum()
    store_colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["success"], PALETTE["danger"]]
    wedges, texts, autotexts = ax.pie(
        by_store.values, labels=by_store.index,
        autopct="%1.1f%%", colors=store_colors,
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold")
    ax.set_title("Sales Distribution by Store")

    plt.tight_layout()
    plt.savefig("outputs/plots/01_eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  EDA plot saved → outputs/plots/01_eda_overview.png")


# ══════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def engineer_features(monthly):
    df = monthly.copy()
    df.sort_values(["store","product","date"], inplace=True)

    grp = df.groupby(["store","product"])["sales_qty"]

    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}m"] = grp.shift(lag)

    # Rolling statistics
    df["rolling_3m_mean"]  = grp.shift(1).rolling(3).mean().reset_index(0, drop=True)
    df["rolling_6m_mean"]  = grp.shift(1).rolling(6).mean().reset_index(0, drop=True)
    df["rolling_3m_std"]   = grp.shift(1).rolling(3).std().reset_index(0, drop=True)
    df["rolling_12m_mean"] = grp.shift(1).rolling(12).mean().reset_index(0, drop=True)

    # Calendar features
    df["month"]    = df["date"].dt.month
    df["quarter"]  = df["date"].dt.quarter
    df["year"]     = df["date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Trend index per group
    df["trend_idx"] = df.groupby(["store","product"]).cumcount()

    # Label encoding
    le_store   = LabelEncoder()
    le_product = LabelEncoder()
    df["store_enc"]   = le_store.fit_transform(df["store"])
    df["product_enc"] = le_product.fit_transform(df["product"])

    # Drop rows with NaN (from lags)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Features engineered  |  Final rows: {len(df):,}  |  Columns: {len(df.columns)}")
    return df, le_store, le_product


# ══════════════════════════════════════════════════════════
# 5. TRAIN / TEST SPLIT + MODEL TRAINING
# ══════════════════════════════════════════════════════════

FEATURE_COLS = [
    "lag_1m","lag_2m","lag_3m","lag_6m","lag_12m",
    "rolling_3m_mean","rolling_6m_mean","rolling_3m_std","rolling_12m_mean",
    "month","quarter","year","month_sin","month_cos",
    "trend_idx","promotion","avg_price",
    "store_enc","product_enc"
]

def split_and_train(df):
    # Temporal split: last 6 months = test
    cutoff = df["date"].max() - pd.DateOffset(months=6)
    train  = df[df["date"] <= cutoff].copy()
    test   = df[df["date"] >  cutoff].copy()

    X_train, y_train = train[FEATURE_COLS], train["sales_qty"]
    X_test,  y_test  = test[FEATURE_COLS],  test["sales_qty"]

    print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")

    models = {
        "Linear Regression":       LinearRegression(),
        "Random Forest":           RandomForestRegressor(n_estimators=200, max_depth=12,
                                                          min_samples_leaf=5, random_state=42,
                                                          n_jobs=-1),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                              learning_rate=0.05, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.maximum(preds, 0)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100

        results[name] = {
            "model":  model,
            "preds":  preds,
            "y_test": y_test.values,
            "MAE":    round(mae, 2),
            "RMSE":   round(rmse, 2),
            "MAPE":   round(mape, 2),
        }
        print(f"  [{name}]  MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.2f}%")

    return results, train, test


# ══════════════════════════════════════════════════════════
# 6. VISUALISE RESULTS
# ══════════════════════════════════════════════════════════

def plot_model_comparison(results):
    names  = list(results.keys())
    maes   = [results[n]["MAE"]  for n in names]
    rmses  = [results[n]["RMSE"] for n in names]
    mapes  = [results[n]["MAPE"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Performance Comparison", fontsize=15,
                 fontweight="bold", color=PALETTE["neutral"])

    colors = [PALETTE["primary"], PALETTE["success"], PALETTE["accent"]]

    for ax, metric, values, label in zip(
        axes,
        ["MAE", "RMSE", "MAPE (%)"],
        [maes, rmses, mapes],
        ["Mean Absolute Error", "Root Mean Square Error", "MAPE (%)"]
    ):
        bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.5)
        best_idx = values.index(min(values))
        bars[best_idx].set_edgecolor(PALETTE["danger"])
        bars[best_idx].set_linewidth(2.5)
        ax.set_title(metric, fontsize=12)
        ax.set_ylabel(label)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(values)*0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=PALETTE["neutral"])
        ax.grid(True, axis="y"); ax.set_ylim(0, max(values)*1.2)
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/plots/02_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Model comparison saved → outputs/plots/02_model_comparison.png")


def plot_predictions_vs_actual(results, test_df):
    best_name = min(results, key=lambda n: results[n]["MAPE"])
    best      = results[best_name]

    # Aggregate to date level
    test_copy = test_df.copy()
    test_copy["predicted"] = best["preds"]
    agg = test_copy.groupby("date").agg(
        actual    = ("sales_qty", "sum"),
        predicted = ("predicted", "sum"),
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f"Prediction vs Actual  [{best_name}]",
                 fontsize=15, fontweight="bold", color=PALETTE["neutral"])

    # Line plot
    ax = axes[0]
    ax.plot(agg["date"], agg["actual"],    color=PALETTE["neutral"],  lw=2,   label="Actual")
    ax.plot(agg["date"], agg["predicted"], color=PALETTE["primary"],  lw=2.5, ls="--", label="Predicted")
    ax.fill_between(agg["date"],
                    agg["actual"], agg["predicted"],
                    alpha=0.12, color=PALETTE["danger"])
    ax.set_title("Actual vs Predicted Monthly Sales (Test Period)")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Units Sold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.legend(fontsize=11); ax.grid(True)

    # Residuals
    ax = axes[1]
    residuals = agg["actual"] - agg["predicted"]
    colors_r  = [PALETTE["success"] if r >= 0 else PALETTE["danger"] for r in residuals]
    ax.bar(agg["date"], residuals, color=colors_r, width=20, edgecolor="white", alpha=0.85)
    ax.axhline(0, color=PALETTE["neutral"], lw=1.5)
    ax.set_title("Residuals (Actual − Predicted)")
    ax.set_xlabel("Date"); ax.set_ylabel("Residual Units")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("outputs/plots/03_predictions_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Predictions plot saved → outputs/plots/03_predictions_vs_actual.png")
    return best_name


def plot_feature_importance(results):
    rf = results["Random Forest"]["model"]
    feat_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    feat_imp = feat_imp.sort_values(ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [PALETTE["accent"] if v >= feat_imp.quantile(0.75)
              else PALETTE["primary"] for v in feat_imp.values]
    bars = ax.barh(feat_imp.index, feat_imp.values, color=colors,
                   edgecolor="white", height=0.65)
    ax.set_title("Feature Importance — Random Forest", fontsize=13,
                 fontweight="bold", color=PALETTE["neutral"])
    ax.set_xlabel("Importance Score")
    for bar, val in zip(bars, feat_imp.values):
        ax.text(val + feat_imp.max()*0.01, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, color=PALETTE["neutral"])
    ax.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig("outputs/plots/04_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Feature importance saved → outputs/plots/04_feature_importance.png")


def plot_product_forecast(results, test_df, monthly):
    best_name = min(results, key=lambda n: results[n]["MAPE"])

    # Pick top product by volume
    top_product = monthly.groupby("product")["sales_qty"].sum().idxmax()

    hist = monthly[monthly["product"] == top_product].groupby("date")["sales_qty"].sum()
    test_copy = test_df.copy()
    test_copy["predicted"] = results[best_name]["preds"]
    fcast = (test_copy[test_copy["product"] == top_product]
             .groupby("date")
             .agg(actual=("sales_qty","sum"), predicted=("predicted","sum")))

    fig, ax = plt.subplots(figsize=(16, 6))
    # Historical
    ax.plot(hist.index, hist.values, color=PALETTE["neutral"], lw=1.5,
            label="Historical Sales")
    # Forecast band
    ax.fill_between(fcast.index,
                    fcast["predicted"]*0.92, fcast["predicted"]*1.08,
                    alpha=0.18, color=PALETTE["primary"], label="Forecast Band (±8%)")
    ax.plot(fcast.index, fcast["predicted"], color=PALETTE["primary"],
            lw=2.5, ls="--", label="Forecast")
    ax.plot(fcast.index, fcast["actual"], color=PALETTE["danger"],
            lw=2, marker="o", ms=5, label="Actual (Test)")

    # Separator line
    split_date = fcast.index.min()
    ax.axvline(split_date, color=PALETTE["accent"], lw=2, ls=":", label="Train/Test Split")

    ax.set_title(f"Sales Forecast — {top_product.replace('_',' ')}", fontsize=13,
                 fontweight="bold", color=PALETTE["neutral"])
    ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.legend(fontsize=10); ax.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/plots/05_product_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Product forecast saved → outputs/plots/05_product_forecast.png  [{top_product}]")


# ══════════════════════════════════════════════════════════
# 7. EXPORT RESULTS JSON (for HTML dashboard)
# ══════════════════════════════════════════════════════════

def export_results(results, monthly):
    summary = {}
    for name, r in results.items():
        summary[name] = {"MAE": r["MAE"], "RMSE": r["RMSE"], "MAPE": r["MAPE"]}

    # Monthly total sales for dashboard chart
    agg = monthly.groupby("date")["sales_qty"].sum().reset_index()
    agg["date"] = agg["date"].dt.strftime("%Y-%m")
    chart_data = agg.to_dict(orient="records")

    # Best model predictions vs actual (aggregated)
    best_name = min(results, key=lambda n: results[n]["MAPE"])
    y_test    = results[best_name]["y_test"]
    preds     = results[best_name]["preds"]
    # Sample 30 points for chart
    idx  = np.linspace(0, len(y_test)-1, min(30, len(y_test)), dtype=int)
    pred_vs_actual = [
        {"index": int(i), "actual": int(y_test[i]), "predicted": int(preds[i])}
        for i in idx
    ]

    payload = {
        "model_results":    summary,
        "best_model":       best_name,
        "monthly_sales":    chart_data,
        "pred_vs_actual":   pred_vs_actual,
        "total_records":    int(len(monthly)),
        "products":         int(monthly["product"].nunique()),
        "stores":           int(monthly["store"].nunique()),
        "date_range":       {
            "start": monthly["date"].min().strftime("%Y-%m"),
            "end":   monthly["date"].max().strftime("%Y-%m"),
        }
    }

    with open("outputs/results/metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("  Metrics JSON saved → outputs/results/metrics.json")
    return payload


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  SALES DEMAND FORECASTING SYSTEM — FMCG")
    print("═"*60)

    print("\n[1/6] Generating synthetic FMCG dataset …")
    raw = generate_dataset()

    print("\n[2/6] Preprocessing …")
    monthly = preprocess(raw)

    print("\n[3/6] Exploratory Data Analysis …")
    run_eda(monthly)

    print("\n[4/6] Feature Engineering …")
    featured, le_store, le_product = engineer_features(monthly)

    print("\n[5/6] Training models …")
    results, train_df, test_df = split_and_train(featured)

    print("\n[6/6] Generating output plots …")
    plot_model_comparison(results)
    best_name = plot_predictions_vs_actual(results, test_df)
    plot_feature_importance(results)
    plot_product_forecast(results, test_df, monthly)
    payload = export_results(results, monthly)

    print("\n" + "═"*60)
    print("  FINAL RESULTS SUMMARY")
    print("═"*60)
    for name, r in results.items():
        marker = " ◀ BEST" if name == best_name else ""
        print(f"  {name:<28}  MAE={r['MAE']:>7.1f}  RMSE={r['RMSE']:>7.1f}  MAPE={r['MAPE']:>5.2f}%{marker}")
    print("═"*60)
    print("\n  All outputs saved to → outputs/")
    print("  Open  outputs/dashboard.html  in your browser for the interactive dashboard.\n")
