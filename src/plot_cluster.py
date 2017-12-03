from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
# setup Lambert Conformal basemap.
# draw coastlines.

region_cluster = consommation_clusters.groupby(by="Region")["Cluster"].value_counts().index.to_frame()
region_cluster.index = region_cluster["Region"].values

region_codes = pd.read_csv("./data/frenchRegions.csv")

region_cluster["Region"].isin(region_codes["Region"])
region_cluster["region_match"] = region_cluster["Region"]
region_cluster.loc[region_cluster["Region"]=="Auvergne-Rhône-Alpes", "region_match"] = [83, 82]
region_cluster.loc[region_cluster["Region"]=="Bourgogne-Franche-Comté", "region_match"] = [26, 43]
region_cluster.loc[region_cluster["Region"]=="Bretagne", "region_match"] = [53]
region_cluster.loc[region_cluster["Region"]=="Centre-Val de Loire", "region_match"] = [24]
region_cluster.loc[region_cluster["Region"]=="Grand-Est", "region_match"] = [42, 21, 41]
region_cluster.loc[region_cluster["Region"]=="Hauts-de-France", "region_match"] = [31, 22]
region_cluster.loc[region_cluster["Region"]=="Ile-de-France", "region_match"] = [11]
region_cluster.loc[region_cluster["Region"]=="Normandie", "region_match"] = [23, 25]
region_cluster.loc[region_cluster["Region"]=="Nouvelle-Aquitaine", "region_match"] = [72, 54, 74]
region_cluster.loc[region_cluster["Region"]=="Occitanie", "region_match"] = [91, 73]
region_cluster.loc[region_cluster["Region"]=="PACA", "region_match"] = [93]
region_cluster.loc[region_cluster["Region"]=="Pays-de-la-loire", "region_match"] = [52]


"""
region_codes = {}
region_codes["Auvergne-Rhône-Alpes"] = [83, 82]
region_codes["Bourgogne-Franche-Comté"] = [26, 43]
region_codes["Bretagne"] = [53]
region_codes["Centre-Val de Loire"] = [24]
region_codes["Grand-Est"] = [42, 21, 41]
region_codes["Hauts-de-France"] = [31, 22]
region_codes["Ile-de-France"] = [11]
region_codes["Normandie"] = [23, 25]
region_codes["Nouvelle-Aquitaine"] = [72, 54, 74]
region_codes["Occitanie"] = [91, 73]
region_codes["PACA"] = [93]
region_codes["Pays-de-la-Loire"] = [52]
"""

plt.show()
import  pygal
from itertools import chain
fr_chart = pygal.maps.fr.Regions()
fr_chart.title = 'Regions clusters'
for cluster in np.unique(region_cluster["Cluster"]):
    fr_chart.add("Cluster " + str(cluster),
                 list(chain.from_iterable([region_codes[region]
                                           for region in region_cluster.loc[
                                               region_cluster["Cluster"]==cluster, "Region"].values])))
fr_chart.render_to_file("test.svg")
fr_chart.render_to_png("test.png")

consommation_30mins = consommation_backup.apply(zscore, axis=0)
plt.figure()
for cluster in np.unique(region_cluster["Cluster"]):
    regions = region_cluster.loc[region_cluster["Cluster"]==cluster, "Region"]
    hourly_consumption = consommation_30mins.loc[:, regions].groupby([consommation_30mins.index.hour]).median().mean(axis=1)
    hourly_consumption.plot(label=cluster)
plt.legend()


plt.figure()
for cluster in np.unique(region_cluster["Cluster"]):
    regions = region_cluster.loc[region_cluster["Cluster"]==cluster, "Region"]
    monthly_consumption = consommation_30mins.loc[:, regions].groupby([consommation_30mins.index.month]).median().mean(axis=1)
    monthly_consumption.plot(label=cluster)
plt.legend()
