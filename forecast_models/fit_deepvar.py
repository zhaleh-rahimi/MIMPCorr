# fir_deepvar.py

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from gluonts.dataset.common import ListDataset
from gluonts.model.deepvar import DeepVAR
from gluonts.mx.trainer import Trainer
from gluonts.evaluation import make_evaluation_predictions

def fit_deepvar(train, test, p, q):
    """
    Fits a DeepVAR model for multivariate one-step-ahead forecasting (like VARMA extend).

    Parameters:
        train (pd.DataFrame): Training data (each column = one time series).
        test (pd.DataFrame): Test data (same columns as train).
        p (int): Number of lags (context length).
        q (int): Prediction length (forecast horizon, keep 1 for one-step ahead).

    Returns:
        predictions (pd.DataFrame): Out-of-sample 1-step-ahead predictions (aligned with test index).
        fitted_values (pd.DataFrame): In-sample fitted values (aligned with train index).
    """
    
    # Ensure datetime index with frequency
    train = train.copy()
    test = test.copy()
    if not isinstance(train.index, pd.DatetimeIndex):
        train.index = pd.date_range(start="2000-01-01", periods=len(train), freq="D")
        test.index = pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=len(test), freq="D")
    freq = pd.infer_freq(train.index) or "D"

    # Helper to convert DataFrame -> GluonTS ListDataset
    def to_listdataset(df):
        return ListDataset(
            [
                {
                    "start": df.index[0],
                    "target": df[col].values
                }
                for col in df.columns
            ],
            freq=freq
        )
    
    # Train dataset
    train_ds = to_listdataset(train)

    # Train model once
    estimator = DeepVAR(
        freq=freq,
        prediction_length=1,   # one-step-ahead forecast
        context_length=p,
        num_layers=2,
        num_cells=40,
        dropout_rate=0.1,
        trainer=Trainer(epochs=5, learning_rate=1e-3, batch_size=16)
    )
    predictor = estimator.train(train_ds)

    # In-sample fitted values
    fitted_pred, _ = make_evaluation_predictions(dataset=train_ds, predictor=predictor, num_eval_samples=1)
    fitted_values = pd.DataFrame([list(pred) for pred in fitted_pred]).T
    fitted_values.index = train.index
    fitted_values.columns = train.columns

    # Out-of-sample rolling one-step-ahead
    rolling_predictions = []
    history = train.copy()

    for t in range(len(test)):
        # Extend available history by one more test point (without labels for the future)
        history_with_future = pd.concat([history, test.iloc[:t+1].copy()])
        dataset_full = to_listdataset(history_with_future)

        # Forecast the next step
        future_pred, _ = make_evaluation_predictions(dataset=dataset_full, predictor=predictor, num_eval_samples=1)
        step_forecast = [list(pred)[-1] for pred in future_pred]  # last point predicted for each series
        rolling_predictions.append(step_forecast)

        # Add actual observation for the next iteration
        history = history_with_future

    predictions = pd.DataFrame(rolling_predictions, index=test.index, columns=test.columns)

    return predictions, fitted_values
