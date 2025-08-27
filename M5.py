# This cell writes a complete, ready-to-run Python script that:
# - Loads M5 sales & calendar CSVs
# - Aggregates daily item sales to weekly per item (optionally pooled across stores)
# - Computes lagged cross-correlations and builds a similarity graph
# - Forms clusters of 2–6 items based on correlation threshold
# - Fits VAR (or VARMAX, if desired) within each cluster
# - Computes rolling order-up-to inventory costs and compares vs. ARIMA-per-item baseline
# - Saves outputs (clusters, costs) to CSV
#
# Place the M5 CSVs in /inputs before running:
#   - /inputs/m5/sales_train_validation.csv
#   - /inputs/m5/calendar.csv
#
# Notes:
# - This is a scalable reference implementation with conservative defaults.
# - For actual VARMA, you can switch model_type="VARMAX" below. VARMAX can be brittle;
#   start with small p,q and short clusters.
# - This code uses diagonal innovation variance for safety-stock (per-item sigma).
# - The evaluation uses a rolling-origin split with periodic refits.
# - You can adjust cost ratios, correlation thresholds, and cluster sizes at the CONFIG block.
#
# -----------------------------------------------------------------------------
# M5 → Weekly → Correlation Clusters → VAR/VARMAX policy vs ARIMA baseline
# -----------------------------------------------------------------------------
# - Robust to sklearn>=1.2 (uses metric='precomputed')
# - Uses SciPy's norm.ppf for fractile
# - VARMAX fitting stabilized (diagonal innovations, multi-optimizer, get_forecast)
# - Always returns mu/sigma (no UnboundLocalError)
# - Guards for short windows, NaNs, scaling
# - Saves clusters/correlations/results/plot to CONFIG["out_dir"]
# -----------------------------------------------------------------------------

import os
import math
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, norm
from sklearn.cluster import AgglomerativeClustering

from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX

# ----------------------------- CONFIG -----------------------------
CONFIG = {
    # File paths
    "sales_csv": "./inputs/m5/sales_train_validation.csv",
    "calendar_csv": "./inputs/m5/calendar.csv",

    # Filtering by department or category (set to None to use all)
    "dept_id": None,   # e.g., 'HOUSEHOLD_1' or None
    "cat_id": "HOBBIES",           # e.g., 'HOBBIES' or None

    # Aggregation
    "pool_across_stores": True,   # sum same item across all stores
    "min_nonzero_ratio": 0.30,    # drop items with too many zeros
    "weekly_freq": "W",           # resample frequency

    # Clustering
    "max_lag_weeks": 8,
    "corr_threshold": 0.30,
    "cluster_size_min": 2,
    "cluster_size_max": 6,
    "n_top_clusters": 30,

    # Modeling
    "model_type": "VAR",       # "VAR" or "VARMAX"
    "p_order": 2,
    "q_order": 0,                 # only used for VARMAX
    "rolling_window_weeks": 104,  # train window length per step
    "test_weeks": 52,             # evaluation horizon (last N weeks)

    # Costs (uniform for simplicity)
    "holding_cost": 1.0,
    "shortage_cost": 1.0,

    # Output directory
    "out_dir": "./outputs/m5"
}

os.makedirs(CONFIG["out_dir"], exist_ok=True)
warnings.filterwarnings("ignore")


# ----------------------------- HELPERS -----------------------------

def load_m5_weekly(config: dict) -> pd.DataFrame:
    """Load M5 sales/calendar, aggregate daily → weekly per item (optionally pooled across stores)."""
    sales = pd.read_csv(config["sales_csv"])
    cal = pd.read_csv(config["calendar_csv"])
    if config["dept_id"] is not None:
        sales = sales[sales["dept_id"] == config["dept_id"]]
    if config["cat_id"] is not None:
        sales = sales[sales["cat_id"] == config["cat_id"]]

    # Pooling key
    key_cols = ["item_id"] if config["pool_across_stores"] else ["item_id", "store_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    # Map day cols to dates
    d_to_date = dict(zip(cal["d"], pd.to_datetime(cal["date"])))
    date_index = [d_to_date[d] for d in day_cols]

    # Build wide matrix: rows = pooled key, cols = dates
    mat = pd.DataFrame(sales[day_cols].values, columns=date_index)
    mat["key"] = sales[key_cols].agg("_".join, axis=1) if len(key_cols) > 1 else sales[key_cols[0]]
    mat = mat.groupby("key").sum()       # sum across pooling level
    # Drop series with too many zeros
    nonzero_ratio = (mat > 0).sum(axis=1) / mat.shape[1]
    mat = mat[nonzero_ratio >= config["min_nonzero_ratio"]]

    # Time index + weekly resample
    mat = mat.T
    mat.index = pd.to_datetime(mat.index)
    weekly = mat.resample(config["weekly_freq"]).sum().sort_index()

    # Drop all-zero series (after resample) and NaNs
    weekly = weekly.loc[:, weekly.sum(axis=0) > 0]
    weekly = weekly.fillna(0.0)
    return weekly  # shape: weeks x items


def standardized_cross_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    """Max abs Pearson correlation across lags in [-max_lag, max_lag]."""
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    best = 0.0
    T = len(x)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = x[-lag:], y[:T + lag]
        elif lag > 0:
            a, b = x[:T - lag], y[lag:]
        else:
            a, b = x, y
        if len(a) > 3:
            c, _ = pearsonr(a, b)
            best = max(best, abs(c))
    return best


def build_clusters(weekly_df: pd.DataFrame, config: dict) -> List[List[str]]:
    """Threshold lagged cross-corr graph → connected components → (optionally) split big comps."""
    items = list(weekly_df.columns)
    n = len(items)

    # Pairwise max-lag correlation matrix
    corr_mat = np.zeros((n, n), dtype=float)
    X = weekly_df.values
    for i in range(n):
        for j in range(i + 1, n):
            c = standardized_cross_corr(X[:, i], X[:, j], max_lag=config["max_lag_weeks"])
            corr_mat[i, j] = corr_mat[j, i] = c

    corr_df = pd.DataFrame(corr_mat, index=items, columns=items)
    corr_df.to_csv(os.path.join(config["out_dir"], "lagged_abs_correlation_matrix.csv"))

    # Adjacency by threshold
    adj = (corr_mat >= config["corr_threshold"]).astype(int)
    np.fill_diagonal(adj, 0)

    # Connected components (DFS)
    visited = set()
    comps = []
    for i in range(n):
        if i in visited:
            continue
        stack, comp = [i], set()
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            nbrs = np.where(adj[u] == 1)[0].tolist()
            stack.extend([v for v in nbrs if v not in visited])
        if len(comp) >= config["cluster_size_min"]:
            comps.append([items[idx] for idx in sorted(comp)])

    # Split large components with hierarchical clustering (distance=1-corr)
    refined = []
    for comp in comps:
        if len(comp) <= config["cluster_size_max"]:
            refined.append(comp)
        else:
            sub = corr_df.loc[comp, comp].values
            dist = np.clip(1.0 - sub, 0.0, None)
            n_sub = math.ceil(len(comp) / config["cluster_size_max"])
            hc = AgglomerativeClustering(n_clusters=n_sub, metric='precomputed', linkage='average')
            labels = hc.fit_predict(dist)
            for lab in np.unique(labels):
                members = [comp[k] for k in range(len(comp)) if labels[k] == lab]
                for i in range(0, len(members), config["cluster_size_max"]):
                    chunk = members[i:i + config["cluster_size_max"]]
                    if len(chunk) >= config["cluster_size_min"]:
                        refined.append(chunk)

    # Rank clusters by internal corr and keep top K
    scored = []
    for grp in refined:
        sub = corr_df.loc[grp, grp].values
        upper = sub[np.triu_indices_from(sub, k=1)]
        score = upper.mean() if upper.size > 0 else 0.0
        scored.append((grp, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    scored = scored[:config["n_top_clusters"]]

    # Save clusters
    rows = []
    for cid, (grp, score) in enumerate(scored, 1):
        rows.extend({"cluster_id": cid, "item": it, "avg_internal_corr": score} for it in grp)
    pd.DataFrame(rows).to_csv(os.path.join(config["out_dir"], "clusters.csv"), index=False)

    return [grp for grp, _ in scored]


def order_up_to_levels(mu: np.ndarray, sigma: np.ndarray, h: float, s: float) -> np.ndarray:
    """ y* = mu + z0 * sigma, z0 at s/(s+h). """
    fractile = s / (s + h)
    z0 = norm.ppf(fractile)
    return mu + z0 * sigma


def safe_one_step_forecast(window_df: pd.DataFrame, p: int, q: int, model_type: str):
    """
    Always returns (mu_vec, sigma_vec) for one-step-ahead forecast.
    Pipeline: standardize → try VARMAX/VAR → forecast mean + 1-step cov (or residual std) → unscale.
    """
    # NaN/inf guard
    W = window_df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').fillna(0.0)

    # Default naive fallback
    last = W.values[-1, :]
    diffs = W.diff().dropna()
    sigma_naive = diffs.std(axis=0).values if not diffs.empty else np.ones(W.shape[1])
    mu_vec = last.copy()
    sigma_vec = np.maximum(sigma_naive, 1e-8)

    # Too short to fit AR parts
    if len(W) <= max(p + 1, 8):
        return mu_vec, sigma_vec

    # Standardize for conditioning
    mean_ = W.mean(axis=0)
    std_ = W.std(axis=0).replace(0.0, 1.0)
    Z = (W - mean_) / std_
    Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    try:
        if model_type.upper() == "VARMAX":
            model = VARMAX(
                Z,
                order=(p, q),
                trend='c',
                enforce_stationarity=True,
                enforce_invertibility=False if q == 0 else True,
                error_cov_type='diagonal'  # more stable than 'unstructured'
            )

            res = None
            for fit_kwargs in (
                dict(method='powell', maxiter=400, disp=False),
                dict(method='bfgs',   maxiter=600, disp=False),
                dict(method='lbfgs',  maxiter=800, disp=False),
            ):
                try:
                    res = model.fit(**fit_kwargs)
                    break
                except Exception:
                    res = None
            if res is None:
                # last attempt with defaults
                res = model.fit(disp=False)

            fc = res.get_forecast(steps=1)
            mu_scaled = fc.predicted_mean.iloc[0].to_numpy(dtype=float)
            try:
                cov = fc.covariance_matrix.iloc[0].to_numpy(dtype=float)
                sigma_scaled = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            except Exception:
                eps = res.resid.to_numpy() if hasattr(res, "resid") else (Z.to_numpy() - res.fittedvalues.to_numpy())
                sigma_scaled = eps.std(axis=0, ddof=1)

        else:  # VAR
            var = VAR(Z).fit(maxlags=p, ic=None, trend='c')
            mu_scaled = var.forecast(y=Z.to_numpy()[-p:], steps=1)[0]
            resid = var.resid.to_numpy() if hasattr(var, "resid") else (Z.to_numpy() - var.fittedvalues.to_numpy())
            sigma_scaled = resid.std(axis=0, ddof=1)

        # Unscale
        mu_vec = mu_scaled * std_.values + mean_.values
        sigma_vec = np.maximum(sigma_scaled * std_.values, 1e-8)
        return mu_vec, sigma_vec

    except Exception:
        # Fallback to VAR
        try:
            var = VAR(Z).fit(maxlags=p, ic=None, trend='c')
            mu_scaled = var.forecast(y=Z.to_numpy()[-p:], steps=1)[0]
            resid = var.resid.to_numpy() if hasattr(var, "resid") else (Z.to_numpy() - var.fittedvalues.to_numpy())
            sigma_scaled = resid.std(axis=0, ddof=1)
            mu_vec = mu_scaled * std_.values + mean_.values
            sigma_vec = np.maximum(sigma_scaled * std_.values, 1e-8)
            return mu_vec, sigma_vec
        except Exception:
            # Naive already set
            return mu_vec, sigma_vec


def run_cluster_policy_cost(weekly_df: pd.DataFrame, items: List[str], config: dict) -> Dict[str, float]:
    """Rolling evaluation of cluster model (VAR/VARMAX) vs per-item ARIMA baseline."""
    df = weekly_df[items].copy().dropna()
    T, n = df.shape

    # Split train/test
    train_T = max(52, T - config["test_weeks"])
    train = df.iloc[:train_T, :]
    test = df.iloc[train_T:, :]
    dates_test = test.index

    h, s = config["holding_cost"], config["shortage_cost"]

    def per_period_cost(y, d):
        over = np.maximum(y - d, 0.0)
        under = np.maximum(d - y, 0.0)
        return h * over.sum() + s * under.sum()

    full = df.to_numpy()
    cost_cluster = 0.0
    cost_arma = 0.0

    for t_idx, _ in enumerate(dates_test):
        end_idx = train_T + t_idx
        start_idx = max(0, end_idx - config["rolling_window_weeks"])
        window = df.iloc[start_idx:end_idx, :]

        # Cluster model forecast (always returns mu,sigma)
        mu_vec, sigma_vec = safe_one_step_forecast(
            window, p=config["p_order"], q=config["q_order"], model_type=config["model_type"]
        )
        y_star = order_up_to_levels(mu_vec, sigma_vec, h, s)
        d_real = full[end_idx, :]
        cost_cluster += per_period_cost(y_star, d_real)

        # Baseline: per-item ARIMA (small order, robust)
        mu_arma = np.zeros(n)
        sigma_arma = np.zeros(n)
        for j in range(n):
            series = window.iloc[:, j]
            try:
                arima = ARIMA(series, order=(config["p_order"], 0, config["q_order"]))
                ar = arima.fit()
                mu_arma[j] = float(ar.forecast(steps=1).iloc[0])
                sigma_arma[j] = float(np.sqrt(max(ar.sigma2, 1e-10)))
            except Exception:
                mu_arma[j] = float(series.iloc[-1])
                sigma_arma[j] = float(series.diff().dropna().std()) if series.size > 1 else 1.0

        y_arma = order_up_to_levels(mu_arma, sigma_arma, h, s)
        cost_arma += per_period_cost(y_arma, d_real)

    return {
        "n_items": n,
        "cost_cluster_model": cost_cluster,
        "cost_baseline_arma": cost_arma,
        "improvement_pct": (cost_arma - cost_cluster) / max(cost_arma, 1e-9) * 100.0
    }


# ----------------------------- MAIN -----------------------------

def main(config: dict):
    print("Loading and aggregating M5 data to weekly...")
    weekly = load_m5_weekly(config)
    print(f"Weekly frame shape: {weekly.shape} (weeks x items)")

    print("Building clusters based on lagged cross-correlations...")
    clusters = build_clusters(weekly, config)
    print(f"Built {len(clusters)} clusters (top {config['n_top_clusters']} kept).")

    results = []
    for cid, grp in enumerate(clusters, 1):
        print(f"Evaluating cluster {cid} with {len(grp)} items... {grp}")
        metrics = run_cluster_policy_cost(weekly, grp, config)
        metrics["cluster_id"] = cid
        results.append(metrics)

    res_df = pd.DataFrame(results)
    res_path = os.path.join(config["out_dir"], "cluster_policy_results.csv")
    res_df.to_csv(res_path, index=False)
    print("Saved results:", res_path)

    # Plot improvement distribution
    plt.figure(figsize=(12, 6))
    plt.hist(res_df["improvement_pct"].values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel("Relative Cost Improvement (%)")
    plt.ylabel("Number of clusters")
    plt.title("Inventory Cost Improvement (Cluster VARMA vs ARMA)")
    plt.tight_layout()
    plot_path = os.path.join(config["out_dir"], "improvement_hist.png")
    plt.savefig(plot_path, dpi=150)
    print("Saved plot:", plot_path)

    # Summary JSON
    summary = {
        "num_clusters": len(clusters),
        "avg_cluster_size": float(np.mean([len(c) for c in clusters])) if clusters else 0.0,
        "median_improvement_pct": float(np.median(res_df["improvement_pct"])) if not res_df.empty else float("nan"),
    }
    pd.Series(summary).to_json(os.path.join(config["out_dir"], "summary.json"), indent=2)
    print("Summary:", summary)


if __name__ == "__main__":
    main(CONFIG)
