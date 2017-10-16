def clean_series(ts):
    ts = remove_outliers(ts)
    ts = ts.interpolate("time")
    return ts


def remove_outliers(ts):
    return ts[abs(ts - ts.rolling('30d').median()) < 2.5 * ts.std()]


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
