import numpy as np
from scipy.interpolate import interp1d

#######################
# Array Manipulations #
#######################

def interpolate(array, length):
    '''
    Interpolates an array to a given length.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        length
    Returns:
        interpolated_array: np.array, shape = (n_instances, length)
    '''
    array_length = len(array)
    return interp1d(np.arange(0, array_length), array)(np.linspace(0.0, array_length-1, length))


########################
# Barycenter Averaging #
########################

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
    interpolated_candidates = [interpolate(arr, length) for arr in X]
    return average_barycenter(interpolated_candidates)

################
# Scoping Maps #
################

utils = {'interpolate': interpolate,
         'interpolated_barycenter': interpolated_average,
         'average_barycenter' : average_barycenter
        }