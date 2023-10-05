# <p align="center"> tsclustering
<div align="center">
<p>
<img alt="GitHub" src="https://img.shields.io/github/license/gellison321/tsclustering">
</p>
</div>

## <p align="center"> A clustering tool for timeseries data with temporal distortions.

### <p align="center">[Install From pip](https://pypi.org/project/tsclustering/)
```
$ pip install tsclustering
```

### <p align="center"> Handling Data with Temporal Distortions

KMeans implemenation using DTW and interpolated averaging. This package is able to efficiently handle arrays of varied length.

### <p align="center"> Interpolated Averaging

To avoid the time complexity of other barycenter averaging techniques, we use interpolated averaging to efficiently compute the barycenters of varied-lengthed arrays. The process is as follows:

1. The mean length of the group, $\mu$, is found.
2. Each timeseries is interpolated to create a vector, $\vec{ts_{l}}$, where $||\vec{ts_{l}}|| = \mu$.
3. The average vector is found as the barycenter $$barycenter = \frac{1}{L}  \sum_{l=1}^{L}\vec{ts_{l}}$$where L is the number of timeseries being averaged and $\vec{ts_{l}} \in{\mathbb{R}^{\mu}}$.

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

```python
from tsclustering import KMeans
import pickle

# Loading Example Data 
with open('./data/sample_data/X.pickle','rb') as file:
    X = pickle.load(file)
with open('./data/sample_data/y.pickle','rb') as file:
    y = pickle.load(file)

# Plotting data
for x in X:
    plt.plot(x, color = 'black');
```
<img alt="GitHub" src="https://github.com/gellison321/tsclustering/blob/main/data/resources/varied-length.png?raw=true" width = 75%; height = auto>

```python
# Clustering with KMeans 
km = KMeans(k_clusters = 3, n_init = 10, max_iter = 100,
            centroids = [], metric = 'dtw', averaging = 'interpolated')
km.fit(X)

# Access the clusters and centroids attributes to plot the data
colors = ['red', 'green', 'blue']
for k in range(km.k_clusters):
    cluster = np.array(X, dtype = object)[np.where(np.array(km.clusters) == k)[0]]
    for arr in cluster:
        plt.plot(arr, color = colors[k])
```
<img alt="GitHub" src="https://github.com/gellison321/tsclustering/blob/main/data/resources/clustered-varied-length.png?raw=true" width = 75%; height = auto>

```python
# Computing Inertia
km.get_inertia()
```
93.27448236700336

```python
from sklearn.metrics import rand_score, adjusted_rand_score
print('Rand Index:', round(rand_score(km.clusters, y),2))
print('Adjusted RI:', round(adjusted_rand_score(km.clusters, y),2))
```
Rand Index: 1.0  
Adjusted RI: 1.0

```python
# Soft clustering returns the distance from each instance to each centroid
km.soft_cluster()
```
array([[3.66707504, 3.43053223, 3.5902464 ],  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [3.26707093, 3.60751793, 3.32326565],  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [3.49872418, 3.53796656, 3.60681567],  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [3.4164345 , 3.3215374 , 3.31998848],  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [3.33290798, 3.69574074, 3.53531107],  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [3.5292556 , 3.27362416, 3.69472868],  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [3.72468091, 3.65222014, 3.70735547],  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [3.73722331, 3.62481453, 3.62434249],  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [3.54864015, 3.66082986, 3.31089306],  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [3.75099114, 4.32067397, 3.81107028]])

```python
# Match an incoming time series array to nearest centroid
print('Clustered Labels:', [km.clusters[0], km.clusters[80]])
print('Predicted Labels:', km.predict([X[0], X[80]]))

```
Clustered Labels: [2, 0]  
Predicted Labels: [2, 0]