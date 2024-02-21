from tsclustering import KMeans
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import rand_score
import time
from tsclustering.metrics import metrics

def timer(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        print(f"Time taken to execute the function: {(end - start):.4f} seconds")
    return wrapper


def load_csv(filename, delimiter=',', skip_header=0, dtype = object):
    
    try:
        data = np.genfromtxt(filename, delimiter=delimiter, skip_header=skip_header, dtype=dtype)
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@timer
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
    kmeans.predict([X[0]])
    kmeans.soft_cluster()

    print(rand_score(kmeans.clusters, y)) 


    # testing parallel kmeans
    kmeans = KMeans(k_clusters=3,
                    max_iter=100, 
                    n_init = 5, 
                    window = 0.9,
                    centroids = []
                    )
    
    kmeans.fit(X, parallel_cores = 2)
    kmeans.predict([X[0]])
    kmeans.soft_cluster()

    print(rand_score(kmeans.clusters, y))


    # Testing sequential kmeans with centroids
    kmeans = KMeans(k_clusters=3,
                    max_iter=100, 
                    n_init = 5, 
                    window = 0.9,
                    centroids = X[:3]
                    )
    
    kmeans.fit(X)
    print(rand_score(kmeans.clusters, y))


    # Testing parallel kmeans with centroids
    kmeans = KMeans(k_clusters=3,
                max_iter=100, 
                n_init = 5, 
                window = 0.9,
                centroids = X[:3]
                )
    
    kmeans.fit(X, parallel_cores = 2)
    print(rand_score(kmeans.clusters, y))

    # testing euclidean distance basic
    print(metrics['euclidean'](X[0][:len(X[1])], X[1]))

    # testing euclidean distance with r
    print(metrics['euclidean'](X[0][:len(X[1])], X[1], r = 2))

    # testing euclidean distance with w
    print(metrics['euclidean'](X[0][:len(X[1])], X[1], w = 0.5))

    # testing euclidean distance with r and w
    print(metrics['euclidean'](X[0][:len(X[1])], X[1], r = 2, w = 0.5))

    # augmenting the data

    X = np.array(X, dtype = object)

    # testing dtw basic

    kmeans = KMeans(k_clusters=3,
                    max_iter=100, 
                    n_init = 5, 
                    window = 0.9,
                    centroids = X[:3]
                    )
        
    kmeans.fit(X)
    print(rand_score(kmeans.clusters, y))

    # Testing parallel kmeans with centroids
    kmeans = KMeans(k_clusters=3,
                    max_iter=100, 
                    n_init = 5, 
                    window = 0.9,
                    centroids = X[:3]
                    )
        
    kmeans.fit(X, parallel_cores = 2)

    print('All tests passed!')

if __name__ == '__main__':
    main()