from tsclustering import *
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import rand_score

def load_csv(filename, delimiter=',', skip_header=0, dtype = object):
    
    try:
        data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header, dtype=dtype)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():

    with open('data/sample_data/x.pickle', 'rb') as f:
        X = pickle.load(f)

    with open('data/sample_data/y.pickle', 'rb') as f:
        y = pickle.load(f)

    y = LabelEncoder().fit_transform(y)

    # testing sequential kmeans
    kmeans = KMeans(k_clusters=3,
                max_iter=100, 
                n_init = 5, 
                window = 0.9,
                centroids = []
                )
    
    kmeans.fit(X)
    assert rand_score(kmeans.clusters, y) == 1.0

    kmeans.predict([X[0]])
    kmeans.soft_cluster()

    # testing parallel kmeans
    kmeans = KMeans(k_clusters=3,
                max_iter=100, 
                n_init = 5, 
                window = 0.9,
                centroids = []
                )
    
    kmeans.fit(X, parallel_cores = 2)
    assert rand_score(kmeans.clusters, y) == 1.0

    kmeans.predict([X[0]])
    kmeans.soft_cluster()

    # Testing sequential kmeans with centroids
    kmeans = KMeans(k_clusters=3,
                max_iter=100, 
                n_init = 5, 
                window = 0.9,
                centroids = X[:3]
                )
    
    kmeans.fit(X)
    assert rand_score(kmeans.clusters, y) == 1.0

    # Testing parallel kmeans with centroids
    kmeans = KMeans(k_clusters=3,
                max_iter=100, 
                n_init = 5, 
                window = 0.9,
                centroids = X[:3]
                )
    
    kmeans.fit(X, parallel_cores = 2)
    assert rand_score(kmeans.clusters, y) == 1.0

    
    print('All tests passed!')

if __name__ == '__main__':
    main()