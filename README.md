# <p align="center"> tsclustering
<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/gellison321/tsclustering">
</p>
</div>

## <p align="center"> A clustering tool for timeseries data with temporal distortions.

### <p align="center"> Handling Data with Temporal Distortions

KMean implemenation using DTW and interpolated averaging. This package is able to efficiently handle arrays of varied length.

<div align="center">
<p>
<img alt="GitHub" src="https://github.com/gellison321/tsclustering/blob/main/data/resources/varied-length.png?raw=true" width = 75%; height = auto>
<img alt="GitHub" src="https://github.com/gellison321/tsclustering/blob/main/data/resources/clustered-varied-length.png?raw=true" width = 75%; height = auto>
</p>
</div>

### <p align="center"> Interpolated Averaging

To avoid the time complexity of other barycenter averaging techniques, we use interpolated averaging to efficiently compute the barycenters of varied-lengthed arrays. The process is as follows:


1. The mean length of the group, $\mu$, is found.
2. Each timeseries is interpolated to create a vector, $\vec{ts_{l}}$, where $||\vec{ts_{l}}|| = \mu$.
3. The average vector is found as the barycenter $$barycenter = \frac{1}{L}  \sum_{l=1}^{L}\vec{ts_{l}}$$where L is the number of timeseries being averaged and $\vec{ts_{l}} \in{\mathbb{R}^{n}}$.

<div align="center">
<p>
<img alt="GitHub" src="https://github.com/gellison321/tsclustering/blob/main/data/resources/barycenter.jpg?raw=true" width = 75%; height = auto>
</p>
</div>

#### Distance Metrics
- Dynamic Time Warping
- Cross Correlation
- Euclidean Distance

#### Dependencies
- Numpy
- SciPy
- TSLearn

##  <p align="center"> IMPLEMENTATION
### <p align="center"> [Full Implementation](https://github.com/gellison321/tsclustering/blob/main/implementation.ipynb)

```python
import pickle
from tsclustering import KMeans

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


