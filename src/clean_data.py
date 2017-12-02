import glob
from os.path import join

import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hcl
from scipy.spatial.distance import squareform
from scipy.stats.mstats import zscore
from statsmodels.tsa.stattools import adfuller, pacf, acf

import src.helpers as helpers

"""
from importlib import reload
reload(helpers)
import PyQt5
import matplotlib.pyplot as plt
%matplotlib qt5
"""

# Read Data
############

data_path = "data"

data = pd.concat([
    pd.read_table(
        file, encoding="iso8859_15", delimiter="\t", engine="python",
        index_col=False).iloc[1:-1, :]
    for file in glob.glob(join(data_path, "*.xls"))
])

# Format Data
#############

data = data.reset_index(drop=True)
data["Consommation"] = pd.to_numeric(data["Consommation"], errors='coerce')
data = data.loc[~data["Consommation"].isnull(), :]
data["Datetime"] = pd.to_datetime(
    (data["Date"] + '_' + data["Heures"]).apply(str), format='%Y-%m-%d_%H:%M')
data["Date"] = pd.to_datetime((data["Date"]).apply(str), format='%Y-%m-%d')
data["Heures"] = pd.to_datetime((data["Heures"]).apply(str), format='%H:%M',
                                infer_datetime_format=False).dt.time

# Reorganise Data
#################
consommation = pd.pivot_table(
    data, values='Consommation', index='Datetime', columns=['Périmètre'])
consommation = consommation.drop('France', axis=1)

# Quality Control
#################

# Good series have more than 50% data
good_series = [
    consommation.iloc[:, i].isnull().sum() / consommation.iloc[:, i].shape[0] <
    0.99 for i in range(consommation.shape[1])
]
consommation = consommation.loc[:, good_series]

# consommation.diff(48).iloc[48:,:].resample("30T").mean().apply(lambda ts: ts.interpolate("linear"), axis=1)
# consommation.resample("30T").mean()
# Need to remove outliers now then resample 1h
start_bad_date = np.unique(np.where(consommation.isnull())[0])[0]
consommation.index[start_bad_date]
consommation = consommation.iloc[:start_bad_date - 1, :]

# consommation = consommation.resample("1H").sum()

# consommation.apply(lambda ts: ts.interpolate("time"), axis=1).isnull().sum().sum()

fig, ax = plt.subplots(4, 3, sharex=True, sharey=True)
i = 0
row = 0
for column in consommation.columns:
    col = i % 3
    consommation[column].plot(ax=ax[row, col])
    i += 1
    if col == 2:
        row += 1

# consommation_1h = consommation.resample("1h").sum()[1:]

consommation_backup = consommation.copy()
consommation = consommation_backup.copy()
# consommation["datetime"] = consommation.index.get_values()
consommation["date"] = consommation.index.date
consommation["time"] = consommation.index.time
consommation = pd.pivot_table(pd.melt(consommation, id_vars=["date", "time"]), index="date", values="value",
                              columns=["Périmètre", "time"])

plt.figure()
ax = plt.gca()
for columns in consommation:
    plt.plot(acf(consommation.loc[:,columns].diff(7)[7:], nlags=100), alpha=0.05, color="black")
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")

# Convert to zscore
consommation = consommation.diff(7)[7:].apply(zscore, axis=0)
consommation = consommation.loc[:, consommation.isnull().sum() == 0]
consommation.index = pd.to_datetime(consommation.iloc[:, 0].index)
consommation = consommation.asfreq("1d")


def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


test_stationarity(consommation.iloc[:, 1])

""""
from statsmodels.tsa.stattools import arma_order_select_ic
arma_order_select_ic(consommation.iloc[:,1]).bic_min_order[0]
k = np.max([arma_order_select_ic(consommation.loc[:,colname]).bic_min_order[0] for colname, col in consommation.iteritems()])
"""
k = np.max([np.where(pacf(consommation.loc[:, colname]) < 0)[0][0] for colname, col in consommation.iteritems()])


DM_GCC = np.zeros((consommation.shape[1], consommation.shape[1]))
for i, j in itertools.combinations(range(consommation.shape[1]), 2):
    DM_GCC[i, j] = DM_GCC[j, i] = 1 - helpers.get_GCC(consommation.iloc[:, i], consommation.iloc[:, j], k)
DM_GCC = pd.DataFrame(DM_GCC, index=consommation.columns, columns=consommation.columns)

# sns.clustermap(consommation, col_linkage=hcl.linkage(squareform(DM_GCC)))
plt.figure()
hcl.dendrogram(hcl.linkage(squareform(DM_GCC), method="average"))

plt.figure()
plt.plot(np.arange(.1, 1.1, .1),
         np.array([
             np.unique(
                 hcl.fcluster(hcl.linkage(squareform(DM_GCC), method="average"), t=t, criterion="distance")).shape[0]
             for t in np.arange(0.1, 1.1, 0.1)]))

hcl.fcluster(hcl.linkage(squareform(DM_GCC), method="average"), t=0.4, criterion="distance")
n_clusters = 5
clusters = hcl.fcluster(hcl.linkage(squareform(DM_GCC), method="average"), t=n_clusters, criterion="maxclust")

from sklearn.decomposition import PCA

pca = PCA(n_components=4)
pca.fit(consommation.T)
pca.explained_variance_ratio_
pca_consommation = pca.fit_transform(consommation.T)

plt.figure()
plt.scatter(pca_consommation[:, 0], pca_consommation[:, 1], c=clusters, cmap=plt.cm.get_cmap('Paired', n_clusters))

type(consommation.columns.values[0])
consommation_clusters = pd.DataFrame(np.transpose([[series[0] for series in consommation.columns.values],
                                                   [series[1] for series in consommation.columns.values],
                                                   list(clusters)]), columns=["Region", "Time", "Cluster"])


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


print_full(consommation_clusters)

from sklearn.cluster import SpectralClustering

clusters = SpectralClustering(n_clusters, affinity="precomputed").fit_predict(DM_GCC)
plt.figure()
plt.scatter(pca_consommation[:, 0], pca_consommation[:, 1], c=clusters, cmap=plt.cm.get_cmap('Paired', n_clusters))

from sklearn.cluster import KMeans

eigen_values, eigen_vectors = np.linalg.eigh(DM_GCC)
clusters = KMeans(n_clusters=n_clusters, init='k-means++').fit_predict(eigen_vectors[:, 2:4])
plt.figure()
plt.scatter(pca_consommation[:, 0], pca_consommation[:, 1], c=clusters, cmap=plt.cm.get_cmap('Paired', n_clusters))

from sklearn.cluster import DBSCAN

DBSCAN().fit_predict(DM_GCC)
DBSCAN(min_samples=4, metric="precomputed").fit_predict(DM_GCC)

from sklearn.cluster import MeanShift

clusters = MeanShift(cluster_all=False).fit_predict(DM_GCC)
plt.figure()
plt.scatter(pca_consommation[:, 0], pca_consommation[:, 1], c=clusters, cmap=plt.cm.get_cmap('Paired', n_clusters))

from sklearn.cluster import Birch

clusters = Birch(n_clusters=n_clusters).fit_predict(DM_GCC)
plt.figure()
plt.scatter(pca_consommation[:, 0], pca_consommation[:, 1], c=clusters, cmap=plt.cm.get_cmap('Paired', n_clusters))

import seaborn as sns

sns.heatmap(DM_GCC)
# Generate a mask for the upper triangle
mask = np.zeros_like(DM_GCC, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(DM_GCC, mask=mask, cmap=cmap,
            square=True)
