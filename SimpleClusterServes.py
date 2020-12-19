from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from PlottingFunction import *

# Create the dataset

deuce = pd.read_pickle(Path.cwd() / 'pickle' / 'Old' / "DeuceServe.plk")
ad = pd.read_pickle(Path.cwd() / 'pickle' / 'Old' / "AdServe.plk")
data = deuce.append(ad)
data.columns = ['x02', 'y02', 'px0', 'py0', 'ox0','oy0']
data_scaled = scale(data)

colour_theme_raw = np.array(['red', 'blue'])
y = np.append(np.ones((1, 100)).astype(np.int64), np.zeros((1, 100)).astype(np.int64))

# Cluster Data

cluster = KMeans(n_clusters=2, random_state=0).fit(data)
cluster_scaled = KMeans(n_clusters=2, random_state=0).fit(data_scaled)

centroids = cluster.cluster_centers_
data["Cluster"] = cluster.labels_
data["Cluster_Scaled"] = cluster_scaled.labels_

# Plot Clusters

plt.figure(1)
plt.subplot(3, 1, 1)
plt.scatter(data.x02, data.y02, c=colour_theme_raw[y], zorder = 2)
plt.scatter(data.px0, data.py0, c=colour_theme_raw[y], zorder = 2)
plt.scatter(data.ox0, data.oy0, c=colour_theme_raw[y], zorder = 2)
plt.title('Separated Data')
createCourt()
plt.subplot(3, 1, 2)
plot1D(data=data, title = "Combined De-identified Data")
plt.subplot(3, 1, 3)
plot1D(data=data, centroids= centroids, title = "Separated Data")
plt.show()