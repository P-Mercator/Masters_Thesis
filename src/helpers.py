import numpy as np


def clean_series(ts):
    ts = remove_outliers(ts)
    ts = ts.interpolate("time")
    return ts


def remove_outliers(ts):
    """
    Remove values where absolute difference with rolling median is more than 2.5 standard deviation
    """
    return ts[abs(ts - ts.rolling('30d').median()) < 2.5 * ts.std()]


def k_matrix(ts, k):
    return np.array([ts[(shift):ts.shape[0] - k + shift] for shift in np.arange(0, k + 1)]).T


def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))


# def corr_matrix(X):
#     return np.cov(X).diagonal() ** (-.5) * np.cov(X) * np.cov(X).diagonal() ** (-.5)

# def corr_matrix(X):
#     I = np.identity(X.shape[0])
#     ones = np.ones((1,X.shape[0]))
#     P = I - ones * ones.T / (X.shape[0])
#     S = np.dot(np.dot(X.T, P), X) / (X.shape[0])
#     D = np.diagonal(S)
#     return D**-0.5 * S * D**-0.5
#
# def corr_matrix(X):
#     S = np.cov(X)
#     D = np.diagonal(S)
#     R = D**-0.5 * S * D**-0.5
#     return D**-0.5 * S * D**-0.5

def get_GCC(ts1, ts2, k):
    Xi = k_matrix(ts1, k)
    Xj = k_matrix(ts2, k)
    Xij = np.concatenate((Xi, Xj), axis=1)
    GCC = 1 - np.linalg.det(np.corrcoef(Xij, rowvar=False) ** (1 / 2 * (k + 1))) / (
        np.linalg.det(np.corrcoef(Xi, rowvar=False) ** (1 / 2 * (k + 1))) \
        * np.linalg.det(np.corrcoef(Xj, rowvar=False) ** (1 / 2 * (k + 1))))
    return GCC


def det(matrix):
    sign, logdet = np.linalg.slogdet(matrix)
    return np.exp(logdet)


def autocorr(self, lag=1):
    """
    Lag-N autocorrelation

    Parameters
    ----------
    lag : int, default 1
        Number of lags to apply before performing autocorrelation.

    Returns
    -------
    autocorr : float
    """
    return self.corr(self.shift(lag))
