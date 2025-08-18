# fir_nbeats.py

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

def fit_nbeats(train, test, p, q):
    """
    Fits an N-BEATS model for multivariate one-step-ahead forecasting (like VARMA extend).

    Parameters:
        train (pd.DataFrame): Training data (columns = series). Index must be datetime or will be generated.
        test (pd.DataFrame): Test data (same columns as train).
        p (int): Number of lags (max_encoder_length).
        q (int): Forecast horizon (prediction length, use 1 for one-step ahead).

    Returns:
        predictions (pd.DataFrame): Out-of-sample 1-step-ahead predictions (aligned with test index).
        fitted_values (pd.DataFrame): In-sample fitted values (aligned with train index).
    """

    # Ensure datetime index
    train = train.copy()
    test = test.copy()
    if not isinstance(train.index, pd.DatetimeIndex):
        train.index = pd.date_range(start="2000-01-01", periods=len(train), freq="D")
        test.index = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=len(test), freq="D")
    freq = pd.infer_freq(train.index) or "D"

    # Convert to long format
    def wide_to_long(df):
        return df.reset_index().melt(id_vars=df.index.name, var_name="series", value_name="value")
    
    train_long = wide_to_long(train)
    test_long = wide_to_long(test)

    # Add time_idx
    full_long = pd.concat([train_long, test_long], ignore_index=True)
    full_long["time_idx"] = full_long.groupby("series").cumcount()

    # Train dataset
    training_cutoff = train_long.groupby("series").size().min() - q
    training = TimeSeriesDataSet(
        full_long[full_long.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="value",
        group_ids=["series"],
        max_encoder_length=p,
        max_prediction_length=1,  # one-step-ahead
        time_varying_unknown_reals=["value"],
        target_normalizer=GroupNormalizer(groups=["series"])
    )

    # Data loader
    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)

    # Define N-BEATS model
    nbeats = NBeats.from_dataset(
        training,
        learning_rate=1e-3,
        log_interval=-1,
        log_val_interval=-1,
        weight_decay=1e-2,
        widths=[256, 256],
        backcast_loss_ratio=0.1,
        loss=QuantileLoss(),
    )

    # Train model
    nbeats.fit(train_loader, max_epochs=5, verbose=False)

    # In-sample fitted values
    fitted_values = []
    for series in train.columns:
        series_data = full_long[full_long.series == series]
        series_loader = training.to_dataloader(predict=True, batch_size=64, num_workers=0)
        preds = nbeats.predict(series_loader, mode="prediction").squeeze(-1).detach().numpy()
        fitted_values.append(preds[:len(train)])
    fitted_values_df = pd.DataFrame(np.array(fitted_values).T, index=train.index, columns=train.columns)

    # Out-of-sample rolling one-step-ahead
    predictions_list = []
    history_long = train_long.copy()

    for t in range(len(test)):
        # Extend available history
        history_long = pd.concat([train_long, test_long.iloc[: (t+1) * len(train.columns)]], ignore_index=True)
        history_long["time_idx"] = history_long.groupby("series").cumcount()

        # Create dataset for prediction
        dataset_pred = TimeSeriesDataSet.from_dataset(training, history_long, stop_randomization=True)
        pred_loader = dataset_pred.to_dataloader(train=False, batch_size=64, num_workers=0)

        preds = nbeats.predict(pred_loader, mode="prediction").squeeze(-1).detach().numpy()
        preds_next = preds[-len(train.columns):]  # last prediction for each series
        predictions_list.append(preds_next)

    predictions_df = pd.DataFrame(predictions_list, index=test.index, columns=test.columns)

    return predictions_df, fitted_values_df
