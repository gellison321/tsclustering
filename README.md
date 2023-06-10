# <p align="center"> tsclustering
<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/gellison321/tsclustering">
</p>
</div>

## <p align="center"> A clustering tool for timeseries data with temporal distortions.

### **FEATURES**

### Handling Data with Temporal Distortions

By using DTW and interpolated averaging, this package is able to efficiently handle arrays of varied length without interpolation or padding.

<div align="center">
<p>
<img alt="GitHub" src="https://github.com/gellison321/tsclustering/blob/main/data/resources/varied-length.png?raw=true" width = 75%; height = auto>
<img alt="GitHub" src="https://github.com/gellison321/tsclustering/blob/main/data/resources/clustered-varied-length.png?raw=true" width = 75%; height = auto>
</p>
</div>

### Interpolated Averaging

To sidestep the time complexity of other barycenter averaging techniques, we use interpolated averaging to efficiently compute the barycenters of varied-lengthed arrays. 

1. Each array is interpolated to a vector of the average length of the group
2. The average vector is taken as the barycenter

<div align="center">
<p>
<img alt="GitHub" src="https://github.com/gellison321/tsclustering/blob/main/data/resources/barycenter.jpeg?raw=true" width = 75%; height = auto>
</p>
</div>

### Distance Metrics
- Dynamic Time Warping
- Cross Correlation
- Euclidean Distance

### **DEPENDENCIES**
- Numpy
- SciPy
- TSLearn

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

# Predict Out-of-Sample Data

km.predict([[]])

```

### [Full Implementation](https://github.com/gellison321/tsclustering/blob/main/implementation.ipynb)
