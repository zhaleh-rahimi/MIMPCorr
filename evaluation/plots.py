# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 01:02:32 2025

@author: ME
"""
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

results_df= pd.read_csv('../outputs/csv/noise_level_sense(100).csv')
# Line plots for Noise Level   
for metric in ["MAPE", "RMSE", "total_cost", "percentage_improvement"]:
    plt.figure()
    sns.lineplot(data=results_df, x="RND", y=metric, hue="p", style="q", marker="o", errorbar=None, palette="deep")
    # plt.title(f"{metric.capitalize()} vs. Noise Level Ratio")
    plt.xlabel("Relative Noise Dispersion")
    plt.ylabel(metric)
    plt.legend(title="(p, q)", loc="best", bbox_to_anchor=(1, 1))
    # plt.grid(False)
    plt.savefig(f'../outputs/figures/{metric}-rnd.pdf', format="pdf")
    plt.show()
