import pandas as pd
import numpy as np
from os.path import join
import glob
from scipy.stats.mstats import zscore
from sklearn.preprocessing import StandardScaler
import src.helpers as helpers

"""
from importlib import reload
import PyQt5
import matplotlib.pyplot as plt
%matplotlib qt5
"""

data_path = "data"

data = pd.concat([
    pd.read_table(
        file, encoding="iso8859_15", delimiter="\t", engine="python",
        index_col=False).iloc[1:-1, :]
    for file in glob.glob(join(data_path, "*.xls"))
])

data = data.reset_index(drop=True)
data["Consommation"] = pd.to_numeric(data["Consommation"], errors='coerce')
data = data.loc[~data["Consommation"].isnull(), :]
data["Datetime"] = pd.to_datetime(
    (data["Date"] + '_' + data["Heures"]).apply(str), format='%Y-%m-%d_%H:%M')

consommation = pd.pivot_table(
    data, values='Consommation', index='Datetime', columns='Périmètre')
consommation = consommation.resample("30T").mean()
consommation = consommation.drop('France', axis=1)

good_series = [
    consommation.iloc[:, i].isnull().sum() / consommation.iloc[:, i].shape[0] <
    0.99 for i in range(consommation.shape[1])
]
consommation = consommation.loc[:, good_series]

consommation = consommation.apply(helpers.clean_series, axis=0).resample('30T').mean()

consommation = consommation.diff().fillna(0).apply(zscore, axis=0)

ts1 = consommation.iloc[:, 0]
ts2 = consommation.iloc[:, 1]

np.array([helpers.crosscorr(ts1, ts2, lag=lag) for lag in np.arange(-10, 10, 1)]).mean()
ts1.shift(5, '1d')
