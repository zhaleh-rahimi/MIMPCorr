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
import os
import math
import warnings
import json
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, norm, normaltest, shapiro
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
    "dept_id": None,
    "cat_id": "HOBBIES",

    # Aggregation
    "pool_across_stores": True,
    "min_weeks_demand": 200,
    "weekly_freq": "W",

    # Clustering
    "max_lag_weeks": 8,
    "corr_threshold": 0.3,
    "cluster_size_min": 2,
    "cluster_size_max": 6,
    "n_top_clusters": 100,

    # Modeling
    "model_type": "VAR",
    "p_order": 2,
    "q_order": 0,
    "rolling_window_weeks": 222,
    "test_weeks": 52,

    # Costs
    "holding_cost": 1.0,
    "shortage_cost": 1.0,

    # Output directory + single-file names
    "out_dir": "./outputs/m5",
    "report_filename": "report.json",
    "plot_hist": "improvement_hist.pdf",
    "plot_scatter": "corr_scatter.pdf",
    "plot_sizes": "cluster_sizes.pdf",
}

os.makedirs(CONFIG["out_dir"], exist_ok=True)
warnings.filterwarnings("ignore")


# ----------------------------- HELPERS -----------------------------

def load_m5_weekly(config: dict) -> pd.DataFrame:
    """
    Load M5 sales/calendar, aggregate daily -> weekly per item (optionally pooled across stores).
    """
    sales = pd.read_csv(config["sales_csv"])
    cal = pd.read_csv(config["calendar_csv"])
    if config.get("dept_id") is not None:
        sales = sales[sales["dept_id"] == config["dept_id"]]
    if config.get("cat_id") is not None:
        sales = sales[sales["cat_id"] == config["cat_id"]]

    # Pooling key
    key_cols = ["item_id"] if config.get("pool_across_stores", True) else ["item_id", "store_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    # Map day cols to dates
    d_to_date = dict(zip(cal["d"], pd.to_datetime(cal["date"])))
    date_index = [d_to_date[d] for d in day_cols]

    # Build wide matrix: rows = pooled key, cols = dates (daily)
    mat = pd.DataFrame(sales[day_cols].values, columns=date_index)
    mat["key"] = sales[key_cols].agg("_".join, axis=1) if len(key_cols) > 1 else sales[key_cols[0]]
    mat = mat.groupby("key").sum()       # sum across pooling level

    # Resample to weekly
    mat = mat.T
    mat.index = pd.to_datetime(mat.index)
    weekly = mat.resample(config.get("weekly_freq", "W")).sum().sort_index()

    # Leading-zero rule:
    T = weekly.shape[0]  # total weeks 
    min_weeks = int(config.get("min_weeks_demand", 104))
    keep_items = []
    dropped_all_zero = []
    dropped_leading = []

    for col in weekly.columns:
        series = weekly[col].to_numpy()  # length T
        # find first non-zero index
        nz_idx = np.argmax(series != 0) if np.any(series != 0) else None
        if nz_idx is None or not np.any(series != 0):
            # all zeros -> drop
            dropped_all_zero.append(col)
            continue
        
        first_nz_idx = int(np.where(series != 0)[0][0])
        remaining_weeks = T - first_nz_idx
        if remaining_weeks >= min_weeks:
            keep_items.append(col)
        else:
            dropped_leading.append(col)

    # Debug prints
    print(f"Total series before filtering: {len(weekly.columns)}")
    print(f" - dropped (all zero): {len(dropped_all_zero)}")
    print(f" - dropped (too many leading zeros , not enough remaining weeks): {len(dropped_leading)}")
    print(f"Kept series after leading-zero filter: {len(keep_items)}")

    # Keep only selected items
    weekly = weekly.loc[:, weekly.columns.isin(keep_items)]

    # Final cleanup: drop any trivial all-zero columns and fill NaNs
    weekly = weekly.loc[:, weekly.sum(axis=0) > 0]
    weekly = weekly.fillna(0.0)
    return weekly  # shape: weeks x items

def standardized_cross_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
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


def build_clusters(weekly_df: pd.DataFrame, config: dict) -> List[Tuple[List[str], float]]:
    items = list(weekly_df.columns)
    n = len(items)
    if n == 0:
        return []
    corr_mat = np.zeros((n, n), dtype=float)
    X = weekly_df.values
    for i in range(n):
        for j in range(i + 1, n):
            c = standardized_cross_corr(X[:, i], X[:, j], max_lag=config["max_lag_weeks"])
            corr_mat[i, j] = corr_mat[j, i] = c

    corr_df = pd.DataFrame(corr_mat, index=items, columns=items)

    # adjacency & connected components
    adj = (corr_mat >= config["corr_threshold"]).astype(int)
    np.fill_diagonal(adj, 0)
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

    # split large comps
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

    # score & keep top K
    scored = []
    for grp in refined:
        sub = corr_df.loc[grp, grp].values
        upper = sub[np.triu_indices_from(sub, k=1)]
        score = float(upper.mean()) if upper.size > 0 else 0.0
        scored.append((grp, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    print(f'Total number of clusters: {len(scored)}')
    scored = scored[:config["n_top_clusters"]]

    return scored


def order_up_to_levels(mu: np.ndarray, sigma: np.ndarray, h: float, s: float) -> np.ndarray:
    fractile = s / (s + h)
    z0 = norm.ppf(fractile)
    return mu + z0 * sigma


def run_total_costs_by_ratio(weekly_df, clusters, config, cost_ratios=[0.1, 1.0, 10.0]):
    results = []
    for ratio in cost_ratios:
        h = 1.0
        s = ratio * h
        total_cluster = 0.0
        total_arma = 0.0
        for grp in clusters:
            metrics = run_cluster_policy_cost(weekly_df, grp, {**config, "holding_cost": h, "shortage_cost": s})
            total_cluster += metrics["cost_cluster_model"]
            total_arma += metrics["cost_baseline_arma"]
        results.append({"s_over_h": ratio,
                        "total_cluster_cost": total_cluster,
                        "total_arma_cost": total_arma,
                        "improvement_pct": (total_arma - total_cluster) / max(total_arma,1e-9)*100.0})
    return pd.DataFrame(results)

def safe_one_step_forecast(window_df: pd.DataFrame, p: int, q: int, model_type: str, exog_window: pd.DataFrame = None, exog_forecast: pd.DataFrame = None):
    """
    Now accepts exog_window (same index as window_df) and exog_forecast (1-row DF for next step).
    Returns (mu_vec, sigma_vec).
    """
    W = window_df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    last = W.values[-1, :]
    diffs = W.diff().dropna()
    sigma_naive = diffs.std(axis=0).values if not diffs.empty else np.ones(W.shape[1])
    mu_vec = last.copy()
    sigma_vec = np.maximum(sigma_naive, 1e-8)

    if len(W) <= max(p + 1, 8):
        return mu_vec, sigma_vec

    mean_ = W.mean(axis=0)
    std_ = W.std(axis=0).replace(0.0, 1.0)
    Z = (W - mean_) / std_
    Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # prepare exog (standardize if needed) - must align rows
    exog_for_fit = None
    exog_for_pred = None
    if exog_window is not None:
        # ensure same index length and order
        exog_for_fit = exog_window.reindex(window_df.index).fillna(0.0)
    if exog_forecast is not None:
        # one-row df, index aligned to forecast date
        exog_for_pred = exog_forecast

    try:
        if model_type.upper() == "VARMAX":
            # statsmodels VARMAX accepts exog for fit and for forecast; we provide exog for fit and exog for forecast via 'exog' and 'exog_future'
            model = VARMAX(
                Z,
                exog=exog_for_fit,
                order=(p, q),
                trend='c',
                enforce_stationarity=True,
                enforce_invertibility=False if q == 0 else True,
                error_cov_type='diagonal'
            )

            res = None
            for fit_kwargs in (
                dict(method='powell', maxiter=400, disp=False),
                dict(method='bfgs', maxiter=600, disp=False),
                dict(method='lbfgs', maxiter=800, disp=False),
            ):
                try:
                    res = model.fit(**fit_kwargs)
                    break
                except Exception:
                    res = None
            if res is None:
                res = model.fit(disp=False)

            if exog_for_pred is not None:
                fc = res.get_forecast(steps=1, exog=exog_for_pred)
            else:
                fc = res.get_forecast(steps=1)

            mu_scaled = fc.predicted_mean.iloc[0].to_numpy(dtype=float)
            try:
                cov = fc.covariance_matrix.iloc[0].to_numpy(dtype=float)
                sigma_scaled = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            except Exception:
                eps = res.resid.to_numpy() if hasattr(res, "resid") else (Z.to_numpy() - res.fittedvalues.to_numpy())
                sigma_scaled = eps.std(axis=0, ddof=1)

        else:
            # plain VAR (no exog)
            var = VAR(Z).fit(maxlags=p, ic=None, trend='c')
            mu_scaled = var.forecast(y=Z.to_numpy()[-p:], steps=1)[0]
            resid = var.resid.to_numpy() if hasattr(var, "resid") else (Z.to_numpy() - var.fittedvalues.to_numpy())
            sigma_scaled = resid.std(axis=0, ddof=1)

        mu_vec = mu_scaled * std_.values + mean_.values
        sigma_vec = np.maximum(sigma_scaled * std_.values, 1e-8)
        return mu_vec, sigma_vec

    except Exception:
        # fallback to VAR
        try:
            var = VAR(Z).fit(maxlags=p, ic=None, trend='c')
            mu_scaled = var.forecast(y=Z.to_numpy()[-p:], steps=1)[0]
            resid = var.resid.to_numpy() if hasattr(var, "resid") else (Z.to_numpy() - var.fittedvalues.to_numpy())
            sigma_scaled = resid.std(axis=0, ddof=1)
            mu_vec = mu_scaled * std_.values + mean_.values
            sigma_vec = np.maximum(sigma_scaled * std_.values, 1e-8)
            return mu_vec, sigma_vec
        except Exception:
            return mu_vec, sigma_vec

def fit_cluster_model_residuals(train_df: pd.DataFrame, p: int, q: int, model_type: str):
    W = train_df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    if len(W) <= max(p + 1, 8):
        return np.empty((0, W.shape[1])), "none"

    mean_ = W.mean(axis=0)
    std_ = W.std(axis=0).replace(0.0, 1.0)
    Z = (W - mean_) / std_
    Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    try:
        if model_type.upper() == "VARMAX":
            model = VARMAX(Z, order=(p, q), trend='c', enforce_stationarity=True,
                           enforce_invertibility=False if q == 0 else True, error_cov_type='diagonal')
            res = model.fit(disp=False)
            resid = res.resid.to_numpy() if hasattr(res, "resid") else (Z.to_numpy() - res.fittedvalues.to_numpy())
        else:
            var = VAR(Z).fit(maxlags=p, ic=None, trend='c')
            resid = var.resid.to_numpy()
        resid_unscaled = resid * std_.values[np.newaxis, :]
        return resid_unscaled, "fitted"
    except Exception:
        return np.empty((0, W.shape[1])), "none"


def run_cluster_policy_cost(weekly_df: pd.DataFrame, items: List[str], config: dict) -> Dict[str, float]:
    df = weekly_df[items].copy().dropna()
    T, n = df.shape
    train_T = max(52, T - config["test_weeks"])
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

       
        # forecast date is the next week's timestamp:
        forecast_index = df.index[end_idx]  # this is the date for the forecast
        


        mu_vec, sigma_vec = safe_one_step_forecast(window, p=config["p_order"], q=config["q_order"], 
                                                   model_type=config["model_type"])
        y_star = order_up_to_levels(mu_vec, sigma_vec, h, s)
        d_real = full[end_idx, :]
        cost_cluster += per_period_cost(y_star, d_real)

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
        "cost_cluster_model": float(cost_cluster),
        "cost_baseline_arma": float(cost_arma),
        "improvement_pct": float((cost_arma - cost_cluster) / max(cost_arma, 1e-9) * 100.0)
    }


def test_residuals_normality(resid_matrix: np.ndarray, alpha: float = 0.05):
    if resid_matrix.size == 0:
        return {"n_series": 0, "n_samples": 0, "n_pass": 0, "n_fail": 0, "pass_frac": float("nan"), "details": []}
    T, n = resid_matrix.shape
    details = []
    n_pass = 0
    for j in range(n):
        x = resid_matrix[:, j]
        x = x[~np.isnan(x)]
        if x.size < 3:
            details.append({"idx": j, "test": "none", "pvalue": None, "n": int(x.size), "normal": False})
            continue
        if x.size >= 8:
            stat, p = normaltest(x)
            test_name = "normaltest (D'Agostino K^2)"
        else:
            stat, p = shapiro(x)
            test_name = "shapiro"
        is_normal = (p >= alpha)
        details.append({"idx": j, "test": test_name, "pvalue": float(p), "n": int(x.size), "normal": bool(is_normal)})
        if is_normal:
            n_pass += 1
    return {"n_series": n, "n_samples": T, "n_pass": n_pass, "n_fail": n - n_pass, "pass_frac": float(n_pass) / n, "details": details}


# ----------------------------- MAIN -----------------------------

def main(config: dict):
    print("Loading and aggregating M5 data to weekly...")
    weekly = load_m5_weekly(config)
    print(f"Weekly frame shape: {weekly.shape} (weeks x items)")
    
    
    print("Building clusters based on lagged cross-correlations...")
    clusters_scored = build_clusters(weekly, config)
    print(f"Built {len(clusters_scored)} clusters (top {config['n_top_clusters']} kept).")

    report = {
        "config": {k: v for k, v in config.items() if k not in ["out_dir"]},
        "n_clusters": len(clusters_scored),
        "clusters": [],
        "aggregate": {}
    }

    results_for_corr = []
    cluster_sizes = []

    for cid, (grp, score) in enumerate(clusters_scored, start=1):
        print(f"\n--- Cluster {cid} ---")
        print(f"Items ({len(grp)}): {grp}")
        metrics = run_cluster_policy_cost(weekly, grp, config)
        metrics["cluster_id"] = cid
        print(f"Cluster size (SKUs): {metrics['n_items']}")
        print(f"Cost (cluster model): {metrics['cost_cluster_model']:.2f}")
        print(f"Cost (per-item ARIMA baseline): {metrics['cost_baseline_arma']:.2f}")
        print(f"Improvement (%): {metrics['improvement_pct']:.2f}%")

        df = weekly[grp].copy().dropna()
        T = df.shape[0]
        train_T = max(52, T - config["test_weeks"])
        train = df.iloc[:train_T, :]
        resid_matrix, fit_status = fit_cluster_model_residuals(train, config["p_order"], config["q_order"], config["model_type"])
        norm_summary = test_residuals_normality(resid_matrix, alpha=0.05)
        print(f"Residual normality: {norm_summary['n_pass']} / {norm_summary['n_series']} series passed (fraction {norm_summary['pass_frac']})")
        print(f"Model fit status for residuals: {fit_status}")

        cluster_entry = {
            "cluster_id": cid,
            "items": grp,
            "n_items": int(metrics["n_items"]),
            "avg_internal_corr": float(score),
            "improvement_pct": float(metrics["improvement_pct"]),
            "cost_cluster_model": float(metrics["cost_cluster_model"]),
            "cost_baseline_arma": float(metrics["cost_baseline_arma"]),
            "residuals_normality": {
                "n_series_tested": int(norm_summary["n_series"]),
                "n_pass": int(norm_summary["n_pass"]),
                "n_fail": int(norm_summary["n_fail"]),
                "pass_frac": float(norm_summary["pass_frac"]),
                "per_series": norm_summary["details"]
            }
        }

        report["clusters"].append(cluster_entry)
        results_for_corr.append((score, metrics["improvement_pct"]))
        cluster_sizes.append(len(grp))

    # Aggregate stats
    if cluster_sizes:
        report["aggregate"]["cluster_count"] = len(cluster_sizes)
        report["aggregate"]["cluster_size_stats"] = {
            "min": int(np.min(cluster_sizes)),
            "median": float(np.median(cluster_sizes)),
            "mean": float(np.mean(cluster_sizes)),
            "max": int(np.max(cluster_sizes)),
            "sizes": cluster_sizes
        }
    else:
        report["aggregate"]["cluster_count"] = 0
        report["aggregate"]["cluster_size_stats"] = {}

    improvements = [c["improvement_pct"] for c in report["clusters"]]
    if improvements:
        report["aggregate"]["improvement_stats"] = {
            "median": float(np.median(improvements)),
            "mean": float(np.mean(improvements)),
            "min": float(np.min(improvements)),
            "max": float(np.max(improvements)),
        }
        print("\nImprovement statistics across clusters:")
        print(json.dumps(report["aggregate"]["improvement_stats"], indent=2))
    else:
        report["aggregate"]["improvement_stats"] = {}

    pass_fracs = [c["residuals_normality"]["pass_frac"] for c in report["clusters"] if not math.isnan(c["residuals_normality"]["pass_frac"])]
    report["aggregate"]["residuals_normality_overview"] = {
        "clusters_tested": len(pass_fracs),
        "avg_frac_pass_normality": float(np.nanmean(pass_fracs)) if pass_fracs else float("nan")
    }
    print("\nResiduals normality overview:")
    print(json.dumps(report["aggregate"]["residuals_normality_overview"], indent=2))

    # Save single JSON report
    out_path = os.path.join(config["out_dir"], config["report_filename"])
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved single JSON report to: {out_path}")

    # ------------------------------
    # Save separate plot files
    # ------------------------------
    # Histogram of improvements
    plt.rcParams.update({
    "font.size": 14,          # base font size
    "axes.labelsize": 16,     # x and y labels
    "axes.titlesize": 18,     # title
    "xtick.labelsize": 14,    # x tick labels
    "ytick.labelsize": 14     # y tick labels
})
    if improvements:
        plt.figure(figsize=(8, 5))
        plt.hist(improvements, bins=20, alpha=0.8, edgecolor='black')
        plt.title("Relative Cost Improvement (%)")
        plt.xlabel("Improvement %")
        plt.ylabel("Number of clusters")
        hist_path = os.path.join(config["out_dir"], config["plot_hist"])
        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"Saved histogram to: {hist_path}")

    
    # Bar: cluster sizes
    if cluster_sizes:
        plt.figure(figsize=(10, 5))
        plt.bar(range(1, len(cluster_sizes) + 1), cluster_sizes)
        plt.xlabel("cluster id (ranked)")
        plt.ylabel("n_items")
        plt.title("Cluster sizes")
        sizes_path = os.path.join(config["out_dir"], config["plot_sizes"])
        plt.tight_layout()
        plt.savefig(sizes_path, dpi=200, format='pdf')
        plt.close()
        print(f"Saved cluster sizes bar chart to: {sizes_path}")

    # Compact per-cluster table to console
    print("\nCompact per-cluster table:")
    header = ["cid", "n_items",  "improvement_pct", "frac_pass_normality"]
    print("\t".join(header))
    for c in report["clusters"]:
        print(f"{c['cluster_id']}\t{c['n_items']}\t{c['improvement_pct']:.2f}\t{c['residuals_normality']['pass_frac']:.3f}")

    print("\nDone.")

    


if __name__ == "__main__":
    main(CONFIG)
