from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def fit_arima(train, test, p=1, q=1):
    """
    Fits ARIMA models independently for each variable in the dataset.
    """
    predictions = pd.DataFrame(index=test.index, columns=test.columns)
    fitted = pd.DataFrame(index=train.index, columns=train.columns)
    for column in train.columns:
        # Fit ARIMA
        model = ARIMA(train[column], order=(p, 0, q))
        # model.initialize_approximate_diffuse()
        fit = model.fit()

        # Predict (1-step ahead in-sample forecast)
        fitted[column] = fit.fittedvalues
        # Predict (1-step ahead out-of-sample forecast)
        res = fit.apply(test[column], refit=True)
        predictions[column] = res.fittedvalues

    return predictions, fitted
