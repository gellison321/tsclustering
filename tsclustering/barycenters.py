import numpy as np
from scipy.interpolate import interp1d


def average_barycenter(X):
    '''
    Computes the arithmetic barycenter of a list of arrays.
    
    Parameters:
        X: array-like, shape = (n_instances, length)
    Returns:
        average_barycenter: np.array, shape = (n_instances, length)
    '''
    return np.mean(X, axis = 0)


def interpolated_average(X):
    '''
    Computes the arithmetic barycenter of a list of arrays after 
    interpolating them to the same length.
    
    Parameters:
        X: array-like, shape = (n_instances, length)
    Returns:
        interpolated_average: np.array, shape = (n_instances, length)
    '''
    length = np.mean(np.array(list(map(len, X))), dtype = int)
    interpolate = lambda arr, l: interp1d(np.linspace(0, 1, len(arr)), arr)(np.linspace(0, 1, l))
    interpolated_candidates = [interpolate(arr, length) for arr in X]
    return average_barycenter(interpolated_candidates)


barycenters = {'interpolated_barycenter': interpolated_average,
               'average_barycenter' : average_barycenter
               }