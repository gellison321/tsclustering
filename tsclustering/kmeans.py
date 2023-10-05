from tsclustering.functions import metrics, np, barycenters
from typing import Optional

class KMeans():
    
    def __init__ (self, n_init = 5, k_clusters = 3, max_iter = 100, centroids = [], metric = 'dtw', averaging = 'interpolated'):
        self.k_clusters = k_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.centroids = centroids
        self.metric = metric
        self.method = averaging
        
    def _assign_clusters(self, X: np.array) -> list[int]:
        '''
        Assigns each instance of X to the nearest centroid.

        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            clusters: array-like, shape = (n_instances, 1)
        '''
        return [np.argmin(np.array([metrics[self.metric](x, centroid)**2 for centroid in self.centroids])) for x in X]
    
    def _initialize_centroids(self, k_centroids: int) -> np.array:
        '''
        Initializes k centroids by randomly selecting k instances of X.
        
        Parameters:
            k_centroids: int, number of centroids to initialize
        Returns:
            centroids: array-like, shape = (k_centroids, length)
        '''
        centroids = [self.X[np.random.randint(0, self.X.shape[0])] for k in range(k_centroids)]
        return np.array(centroids, dtype = self.dtype)

    def _update_centroids(self, X: np.array) -> np.array:
        '''
        Updates the centroids by computing the barycenter of each cluster.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            new_centroids: array-like, shape = (k_centroids, length)
        '''
        new_centroids = []
        for k in range(len(self.centroids)):  
            cluster = X[np.where(self.clusters==k)[0]]
            if cluster.shape[0] == 0:
                new_centroids.append(self.centroids[k])
            elif cluster.shape[0] == 1:
                new_centroids.append(cluster[0])
            else:
                new_centroids.append(barycenters[self.method](cluster))
        return np.array(new_centroids, dtype = self.dtype)

    def _check_solution(self, new_centroids: np.array) -> bool:
        '''
        Checks if the solution has converged by checking whether the new centroids 
        are equal to the old centroids.
        
        Parameters:
            new_centroids: array-like, shape = (k_centroids, length)
        Returns:
            bool
        '''
        return np.all([np.array_equal(self.centroids[i], new_centroids[i]) for i in range(len(self.centroids))])
    
    def _get_inertia(self):
        return sum([metrics[self.metric](self.X[i], self.centroids[self.clusters[i]])**2 for i in range(len(self.X))])

    def local_kmeans(self) -> None:
        '''
        Solves the local cluster problem according to Lloyd's algorithm.
        '''
        if len(self.centroids) < self.k_clusters:
            self.centroids = self._initialize_centroids(self.k_clusters)
        for i in range(self.max_iter):
            self.clusters = self._assign_clusters(self.X)
            new_centroids = self._update_centroids(self.X)
            if self._check_solution(new_centroids):
                break
            else:
                self.centroids = new_centroids
        self.inertia = self._get_inertia()

<<<<<<< HEAD
    def sample_kmeans(self) -> None:
        '''
        Solves the global cluster problem by sampling the local cluster problem n_init times.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        '''
=======
    # solves the local cluster problem n_init times and saves the result with the lowest inertia
    def sample_kmeans(self):
>>>>>>> 99d795d4c92802b8b8566943caef8c97d40743e9
        cost = None
        clusters = None
        centroids = None
        for n in range(self.n_init):
            self.centroids = []
            self.clusters = []
            self.local_kmeans()
            if cost is None or self.inertia < cost:
                cost = self.inertia
                clusters = self.clusters
                centroids = self.centroids
        self.inertia = cost
        self.clusters = clusters
        self.centroids = centroids

    def fit(self, X) -> None:
        '''
        Checks the type of data for varied length arrays, and calls the sample_kmeans method.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            clusters: array-like, shape = (n_instances, 1)
        '''
        self.dtype = object if np.any(np.diff(list(map(len, X)))!=0) else 'float64'
        self.X = np.array(X, dtype = self.dtype)
        self.sample_kmeans()

    # Assigns an out of sample X to each of the nearest centroids
    def predict(self, X: np.array) -> list:
        '''
        Assigns each instance of X to the nearest centroid.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            clusters: array-like, shape = (n_instances, 1)
        '''
        dtype = object if np.any(np.diff(list(map(len, X)))!=0) else 'float64'
        X = np.array(X, dtype = dtype)
        return self._assign_clusters(X)

    # Computes the distance of each instance of X to each centroid
    def soft_cluster(self) -> np.array:
        '''
        Computes the distance of each instance of X to each centroid.

        Parameters:
            None
        Returns:
            soft_clusters: array-like, shape = (n_instances, k_centroids)
        '''
        soft_clusters = []
        for centroid in self.centroids:
            distances = []
            for i in range(len(self.X)):
                distances.append(metrics[self.metric](self.X[i], centroid))
            soft_clusters.append(distances)
        a = np.array(soft_clusters)
        a = a.reshape(a.shape[1], a.shape[0]);
        return a