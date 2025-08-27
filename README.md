# MIMPCorr: Multi-Item Multi-Period Correlated-aware Inventory Optimization


## Abstract

This is a study on multi-item periodic-review inventory systems,
considering stochastic demand with different types of correlations, including auto- and cross-correlation.
We find the VARMA(p,q) models to be suitable structural frameworks to integrate dependencies for a tractable
analysis of optimal ordering policies. Through experimental studies, we evaluate inventory
costs and cost improvements compared to multi-item ordering policies where demands are
assumed to be independent under different degrees of correlation, noise levels, and training
data window sizes. The results show that, for moderate to high levels of dependence among
products, the proposed framework can meaningfully decrease inventory costs.

## Experimental Setup 
The goal is to validate the theoretical findings by evaluating the optimal ordering policy
derived from our multi-item inventory model. The key questions we seek to answer are:

1. How does introducing demand dependence (through VARMA models) instead of assuming independent
demands affect optimal ordering decisions?
2. To what extent does modeling cross-product and auto-correlation impact inventory cost and order
variability compared to independent demand assumptions?
3. How does estimation error in VARMA model parameters affect the accuracy of optimal ordering
decisions and overall inventory performance?
4. Is there sensitivity to cross-correlation strength, noise level, and the amount of training data when
estimating parameters?
5. How well does the proposed optimization framework perform in a real-world setting, compared to
synthetic test cases and an assumption of independence?

To address these questions, we conduct controlled experiments on synthetic demand data, allowing
us to isolate and analyze the effect of demand dependencies in an idealized setting.

## Project Structure

```
mimpio/

├── data_prep                 # synthetic data generation using VARMA process, and data preparation for other datasets
├── evaluation                # scripts to evaluate models
├── forecast_models           # forecast models VARMA and ARIMA 
├── inventory_models          # implementation of proposed inventory optimization model
├── inputs                    # model configs and input data
├── outputs                   # outputs
├── util                      # helper and stat test scripts
├── synthetic.py              # main script to run project on synthetic data
├── M5.py                     # script for empirical study on scalability
```



