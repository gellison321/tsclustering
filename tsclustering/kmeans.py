from tsclustering.utils import np, utils
from tsclustering.dtw import dtw
import multiprocessing

class KMeans():
    
    def __init__ (self, 
                  n_init = 5, 
                  k_clusters = 3, 
                  max_iter = 100, 
                  centroids = [],
                  window = 0.9
                  ):
        self.k_clusters = k_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.centroids = centroids
        self.metric = dtw
        self.method = 'interpolated_barycenter'

        if type(window) in [float, np.float64, np.float32, np.float16, int, np.int64, np.int32, np.int16, np.int8]:
            if window < 0.3:
                print('Warning: too small of a window parameter may lead to insufficient alignment of arrays, and thus to inaccurate results.')
            self.window = window
        elif type(window) == int:
            if window > 1:
                self.window = 1
        else:
            raise TypeError('window must be a float or an int')
        
    def _assign_clusters(self, X, centroids):
        '''
        Assigns each instance of X to the nearest centroid. Enumerates inertia, 
        upon assignment of cluster, to avoid recomputing it.

        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            clusters: array-like, shape = (n_instances, 1)
        '''
        clusters = []
        inertia = 0
        for x in X:
            best_dist = np.inf
            best_cluster = None
            for k in range(len(centroids)):
                dist = self.metric(x, centroids[k], w = self.window, r = best_dist)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = k
            clusters.append(best_cluster)
            inertia += best_dist
        return np.array(clusters), inertia
    
    def _initialize_centroids(self, k_centroids):
        '''
        Initializes k centroids by randomly selecting k instances of X.
        
        Parameters:
            k_centroids: int, number of centroids to initialize
        Returns:
            centroids: array-like, shape = (k_centroids, length)
        '''
        centroids = [self.X[np.random.randint(0, self.X.shape[0])] for _ in range(k_centroids)]
        return np.array(centroids, dtype = self.dtype)

    def _update_centroids(self, X, centroids, clusters):
        '''
        Updates the centroids by computing the barycenter of each cluster.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            new_centroids: array-like, shape = (k_centroids, length)
        '''
        new_centroids = []
        for k in range(len(centroids)):  
            cluster = X[np.where(clusters==k)[0]]
            if cluster.shape[0] == 0:
                new_centroids.append(centroids[k])
            elif cluster.shape[0] == 1:
                new_centroids.append(cluster[0])
            else:
                new_centroids.append(utils[self.method](cluster))
        return np.array(new_centroids, dtype = self.dtype)

    def _check_solution(self, new_centroids, old_centroids):
        '''
        Checks if the solution has converged by checking whether the new centroids 
        are equal to the old centroids.
        
        Parameters:
            new_centroids: array-like, shape = (k_centroids, length)
        Returns:
            bool
        '''
        return all([np.array_equal(old_centroids[i], new_centroids[i]) for i in range(len(old_centroids))])

    def local_kmeans(self, i = None):
        '''
        Solves the local cluster problem according to Lloyd's algorithm.
        '''
        clusters = []
        centroids = self._initialize_centroids(self.k_clusters)
        for _ in range(self.max_iter):
            clusters, inertia = self._assign_clusters(self.X, centroids)
            new_centroids = self._update_centroids(self.X, centroids, clusters)
            if self._check_solution(new_centroids, centroids):
                break
            else:
                centroids = new_centroids

        return clusters, centroids, inertia

    def sample_kmeans(self):
        '''
        Solves the global cluster problem by sampling the local cluster problem n_init times.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        '''
        best_cost = None
        best_clusters = None
        best_centroids = None

        for _ in range(self.n_init):
            clusters, centroids, inertia = self.local_kmeans()
            if best_cost is None or inertia < best_cost:
                best_cost = inertia
                best_clusters = clusters
                best_centroids = centroids

        self.inertia = best_cost
        self.clusters = best_clusters
        self.centroids = best_centroids

    def sample_kmeans_parallel(self, cores = 'auto'):

        num_cpus = multiprocessing.cpu_count()

        if cores == 'auto':
            pool_size = max(1, num_cpus - 1)
        else:
            if cores <= num_cpus:
                pool_size = cores

        with multiprocessing.Pool(pool_size) as pool:
            results = pool.map(self.local_kmeans, range(self.n_init))

        best_cost = None
        best_clusters = None
        best_centroids = None

        for clusters, centroids, inertia in results:
            if best_cost is None or inertia < best_cost:
                best_cost = inertia
                best_clusters = clusters
                best_centroids = centroids

        self.inertia = best_cost
        self.clusters = best_clusters
        self.centroids = best_centroids

    def fit(self, X, cores = 1):
        '''
        Checks the type of data for varied length arrays, and calls the sample_kmeans method.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            clusters: array-like, shape = (n_instances, 1)
        '''
        self.dtype = object if np.any(np.diff(list(map(len, X)))!=0) else 'float64'
        self.X = np.array(X, dtype = self.dtype)
        if self.dtype == 'float64':
            self.method = 'average_barycenter'

        if cores != 1:
            assert type(cores) == int, 'cores must be an integer'
            self.sample_kmeans_parallel(cores = cores)
        else:
            self.sample_kmeans()

    def predict(self, X):
        '''
        Assigns each instance of X to the nearest centroid.
        
        Parameters:
            X: array-like, shape = (n_instances, length)
        Returns:
            clusters: array-like, shape = (n_instances, 1)
        '''
        self.dtype = object if np.any(np.diff(list(map(len, X)))!=0) else 'float64'
        self.X = np.array(X, dtype = self.dtype)
        if self.dtype == 'float64':
            self.method = 'average_barycenter'

        clusters, inertia = self._assign_clusters(X)
        return clusters

    def soft_cluster(self):
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
                distances.append(self.metric(self.X[i], centroid))
            soft_clusters.append(distances)
        a = np.array(soft_clusters)
        a = a.reshape(a.shape[1], a.shape[0]);
        return a