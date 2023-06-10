# <p align="center"> tsclustering
<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/gellison321/timeseriespy">
</p>
</div>

## <p align="center"> A clustering tool for timeseries data with temporal distortions.

### **FEATURES**

### Handling Data with Temporal Distortions

By using DTW (sklearn) and interpolated averaging, this package is able to efficiently handle arrays of varied length without interpolation or padding.

<div align="center">
<p>
<img alt="GitHub" src="https://github.com/gellison321/tsclustering/blob/main/data/resources/varied-length.png?raw=true">
</p>
</div>

<div align="center">
<p>
<img alt="GitHub" src="https://github.com/gellison321/tsclustering/blob/main/data/resources/clustered-tvaried-length.png?raw=true">
</p>
</div>


### Interpolated Averaging

To sidestep the time complexity of other barycenter averaging techniques, we use interpolated averaging to efficiently compute the barycenters of varied-lengthed arrays. 

1. Each array is interpolated to a vector of the average length of the group
2. The average vector is taken as the barycenter


### Distance Metrics
- Dynamic Time Warping
- Cross Correlation
- Euclidean Distance

### **DEPENDENCIES**
- Numpy
- SciPy
- SKLearn

##  <p align="center"> IMPLEMENTATION

```python
import pickle
from tsclustering.kmeans import KMeans

# Loading Example Data 

with open('./data/sample_data/X.pickle','rb') as file:
    X = pickle.load(file)
with open('./data/sample_data/y.pickle','rb') as file:
    y = pickle.load(file)


# Clustering with KMeans 

km = KMeans(k_clusters = 3, n_init = 10, max_iter = 100,
            centroids = [], metric = 'dtw', averaging = 'interpolated')
km.fit(X)

# Computing Inertia

km.get_inertia()

# Soft Clustering

km.soft_cluster()

```

### [Full Implementation](https://github.com/gellison321/tsclustering/blob/main/implementation.ipynb)
