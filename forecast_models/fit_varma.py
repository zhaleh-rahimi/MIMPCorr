from statsmodels.tsa.statespace.varmax import VARMAX
import warnings
warnings.filterwarnings("ignore")


def fit_varma(train, test, p, q):
    """
    Fits a VARMA(p, q) model to the given data.

    Parameters:
        data (pd.DataFrame): Time series data.
        p (int): Number of AR lags.
        q (int): Number of MA lags.

    Returns:
        VARMAX: Fitted VARMA model.
    """
    model = VARMAX(train, order=(p, q))
    fitted_model = model.fit(disp=False,method='powell')

    # Predict (1-step ahead in-sample forecast)
    fitted_values = fitted_model.fittedvalues
    # Predict (1-step ahead out-of-sample forecast)
    res = fitted_model.apply(test, refit=True)
    predictions = res.fittedvalues

    return predictions, fitted_values
