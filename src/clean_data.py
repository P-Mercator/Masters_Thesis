import pandas as pd
import numpy as np
from os.path import join
import glob
"""
import PyQt5
import matplotlib.pyplot as plt
%matplotlib qt5
"""

data_path = "../data"
data = pd.concat([
    pd.read_table(
        file,
        encoding="iso8859_15",
        delimiter="\t",
        index_col=False).iloc[:-1, :]
    for file in glob.glob(join(data_path, "*.xls"))
])


data = data.reset_index(drop=True)
data["Consommation"] = pd.to_numeric(data["Consommation"], errors='coerce')
data = data.loc[~data["Consommation"].isnull(), :]
data["Datetime"] = pd.to_datetime((data["Date"]+ '_' +data["Heures"]).apply(str),format='%Y-%m-%d_%H:%M')

consommation = pd.pivot_table(data, values='Consommation', index='Datetime', columns='Périmètre')
consommation = consommation.drop('France', axis=1)


