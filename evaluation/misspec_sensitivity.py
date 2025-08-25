# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:24:21 2025

@author: Zhaleh
"""

from data_prep.generate_varma_process import varma_data_generator
import pandas as pd
from joblib import Parallel, delayed
import time
from evaluation.evaluate_varma_order_policy import evaluate_varma_order_policy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.linalg import LinAlgError


def misspec_sensitivity_single_run(run_id):

    sigma_base = 1
    min_y = 10
    T= 500
    test_size=100
    train_size = T -test_size
    true_p=4
    true_q=1
    # # Uncomment to run for 2 items
    k = 2
    cost_params = {
        "holding_cost": [[10, 10]],
        "shortage_cost": [[50, 50]]
    }
    max_rho = 0.73
    alpha = 0.29
    # size_list=[30, 80, 100, 200, 300, 400, 500]
    
    # Uncomment to run for 4 items
    # k = 4
    # cost_params = {
    #     "holding_cost": [[10, 10, 10, 10]],
    #     "shortage_cost": [[10, 10, 10,  10]]
    # }
    # max_rho = {(1, 0): 0.8, (1, 1): 0.8, (2, 0): 0.7, (2, 1): 0.7,
    #            (3, 0): 0.7, (3, 1): 0.7, (4, 0): 0.65, (4, 1): 0.65}
    # alpha = {(1, 0): 0.3, (1, 1): 0.3, (2, 0): 0.89, (2, 1): 0.89,
    #          (3, 0): 0.78, (3, 1): 0.78, (4, 0): 0.7, (4, 1): 0.7}
    # size_list=[80, 100, 200, 300, 400, 1000,2000]
    
    improvement_results = []
    config = {
                "time_steps": T,
                "num_products": k,
                "model_order": [true_p, true_q],
                "min_demand": min_y,
                "max_rho": max_rho,
                "alpha": alpha,
                "train_size": train_size,
                "test_size": test_size
            }
    # Simulate data
    varma_generator = varma_data_generator(config=config, seed=run_id)
    data_fit, data_gen = varma_generator.generate_scenarios()
    title = f'Items={k}, p={true_p}, q={true_q}, High Dependence'
    df = {title: data_fit[title]}
    # Iterate    
    for p in range(1, true_p+1):  # p from 1 to 3
        for q in range(true_q+1):        

            # Iterate over cost items
            for cost_idx in range(len(cost_params['holding_cost'])):
                try:
                    costs = {key: values[cost_idx]
                                for key, values in cost_params.items() if len(values) > cost_idx}
                    percentage_improvement, cost, _, forecast_performance,h_cost,s_cost = evaluate_varma_order_policy(
                        df, costs, [p, q], data_gen, min_y, train_size,test_size)

                    improvement_results.append([k,train_size, true_p, true_q , p, q,
                                                np.mean(
                                                    forecast_performance["VARMA"][title]['mape']),
                                                np.mean(
                                                    forecast_performance["VARMA"][title]['rmse']),
                                                cost["VARMA"][title],
                                                percentage_improvement[title]])
                except LinAlgError:
                    print("LU decomposition error occurred! Skipping this iteration and continuing.")
    improvement_results = pd.DataFrame(improvement_results, columns=["Items","Train_Size", "true_p", "true_q","p", "q", "MAPE", "RMSE",
                                                                     "total_cost", "percentage_improvement"])
    plot_error(improvement_results)

    return improvement_results

# batch run


def misspec_batch_run(n_run):
    # Start timer
    start_time = time.time()
    # Run the tasks in parallel
    results = Parallel(n_jobs=-1)(delayed(misspec_sensitivity_single_run)(run_id)
                                  for run_id in range(n_run))

    # End timer
    end_time = time.time()

    # Flatten the list of results
    improvement_results = pd.concat(results, ignore_index=True)

    # # List of columns to average
    # columns_to_average = ["MAPE", "RMSE", "total_cost", "percentage_improvement"]

    # # Compute the average for each column
    # average_results = improvement_results[columns_to_average].mean()
    # average_results= pd.DataFrame(improvement_results, columns=["Train_Size", "p", "q", "MAPE", "RMSE",
    #                                                                  "total_cost","percentage_improvement"])
    # write summary table to an output file
    filename = f"misspec_sens({n_run}).csv"
    summary_tbl_path = f"outputs/csv/{filename}"
    improvement_results.to_csv(summary_tbl_path)

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.5f} seconds")

    plot_error(improvement_results)

# plot error against traning size


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_error(df, outdir="outputs/figures"):
    os.makedirs(outdir, exist_ok=True)

    # Bigger, consistent fonts suited for small subfigures
    mpl.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 200,     # for on-screen; we'll override dpi on save
        "pdf.fonttype": 42,    # embed TrueType for better PDF compatibility
        "ps.fonttype": 42,
    })

    k = df["Items"].iloc[0]
    metrics = ["MAPE", "RMSE", "total_cost", "percentage_improvement"]

    # Use diverging colormap for +/− metrics, sequential otherwise
    cmaps = {
        "percentage_improvement": "RdBu_r",  # centered at 0
        "MAPE": "YlGn",
        "RMSE": "YlGn",
        "total_cost": "YlGn",
    }
    # Per-metric number formatting
    formats = {
        "MAPE": ".0f",
        "RMSE": ".0f",
        "total_cost": ".0f",
        "percentage_improvement": ".0f",
    }

    for metric in metrics:
        agg = (
            df.groupby(["p", "q"], as_index=False)
              .agg(mean=(metric, "mean"))
        )

        # Pivot to p×q grid, sort both axes for a clean layout
        pivot = agg.pivot(index="p", columns="q", values="mean")
        pivot = pivot.sort_index().sort_index(axis=1)
        data = pivot.to_numpy()

        # Figure sized for small subfigures (we export at high dpi)
        fig, ax = plt.subplots(figsize=(3.4, 2.9), dpi=300, constrained_layout=True)

        cmap = mpl.cm.get_cmap(cmaps.get(metric, "YlGn"))
        if metric == "percentage_improvement":
            # Center color scale at 0
            vmax = np.nanmax(np.abs(data))
            norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

        # Ticks / labels
        ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns.tolist())
        ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index.tolist())
        ax.set_xlabel("q")
        ax.set_ylabel("p")
        # Uncomment if you want a title when viewed standalone
        # ax.set_title(f"Mean {metric} by (p, q) · Items={k}")

        # Thin frame
        for s in ax.spines.values():
            s.set_linewidth(0.8)

        # Light grid between cells for legibility
        ax.set_xticks(np.arange(-.5, pivot.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, pivot.shape[0], 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.4, alpha=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Contrast-aware annotations so numbers stay readable on any color
        fmt = formats.get(metric, ".0f")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    txt, color = "–", "black"
                else:
                    rgba = cmap(norm(v))
                    lum = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
                    color = "black" if lum > 0.6 else "white"
                    txt = format(v, fmt)
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, fontweight="semibold", color=color)

        # Colorbar with label
        # cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.04)
        # cbar.ax.tick_params(labelsize=10)
        # cbar.set_label(f"Mean {metric}", fontsize=11)

        # Save high-res PNG (for quick use) and PDF (crisp in LaTeX)
        stem = f"misspec-{metric}"
        png = os.path.join(outdir, stem + ".png")
        pdf = os.path.join(outdir, stem + ".pdf")
        # fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
        plt.show()
        plt.close(fig)
   
