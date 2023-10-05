import numpy as np
from tslearn.metrics import dtw
from scipy.interpolate import interp1d
from scipy.signal import correlate
from typing import Optional

#######################
# Array Manipulations #
#######################

def interpolate(array: Optional[np.array], length: int) -> np.array:
    '''
    Interpolates an array to a given length.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        length: int
    Returns:
        interpolated_array: np.array, shape = (n_instances, length)
    '''
    if type(array) != np.array:
        array = np.array(array)
    array_length = len(array)
    return interp1d(np.arange(0, array_length), array)(np.linspace(0.0, array_length-1, length))

def reinterpolate(array: Optional[np.array], window_length: int) -> np.array:
    '''
    Repeats an array until it reaches a given length.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        window_length: int
    Returns:
        reinterpolated_array: np.array, shape = (n_instances, window_length)
    '''
    if type(array) != np.array:
        array = np.array(array)
    length = len(array)
    return np.concatenate([np.tile(array, window_length//length),array[:window_length%length]])

def pad(array: Optional[np.array], length: int) -> np.array:
    '''
    Pads an array with zeros until it reaches a given length.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        length: int
    Returns:
        padded_array: np.array, shape = (n_instances, length)
    '''
    if type(array) != np.array:
        array = np.array(array)
    return np.pad(array, (0,length - len(array)), 'constant')

def center_moving_average(array: Optional[np.array], period: int) -> np.array:
    '''
    Computes the centered moving average of an array.
    
    Parameters:
        array: array-like, shape = (n_instances, length)
        period: int
    Returns:
        centered_moving_average: np.array, shape = (n_instances, length - period + 1)
    '''
    if type(array) != np.array:
        array = np.array(array)
    ret = np.cumsum(array)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

####################
# Distance Metrics #
####################

def cross_correlation(arr1: Optional[np.array], arr2: Optional[np.array], method: str = 'avg') -> float:

    if type(arr1) != np.array:
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
    cases = {'avg': np.mean, 'max' : np.max, 'min' : np.min}
    return cases[method](correlate(arr1, arr2))

def euclidean_distance(arr1: Optional[np.array], arr2: Optional[np.array]) -> float:
    '''
    Computes the euclidean distance between two arrays.
    
    Parameters:
        arr1: array-like, shape = (n_instances, length)
        arr2: array-like, shape = (n_instances, length)
    
    Returns:
        euclidean_distance: float
    '''
    if type(arr1) != np.array:
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
    return np.linalg.norm(arr1-arr2)

def dynamic_time_warping(arr1: Optional[np.array], arr2: Optional[np.array]) -> float:
    '''
    Implements the dynamic time warping algorithm.
    
    Parameters:
        arr1: array-like, shape = (n_instances, length)
        arr2: array-like, shape = (n_instances, length)
    Returns:
        dtw: float
    '''
    return dtw(arr1, arr2)

########################
# Barycenter Averaging #
########################

def average_barycenter(X: Optional[np.array]) -> np.array:
    '''
    Computes the arithmetic barycenter of a list of arrays.
    
    Parameters:
        X: array-like, shape = (n_instances, length)
    Returns:
        average_barycenter: np.array, shape = (n_instances, length)
    '''
    if type(X) != np.ndarray:
        X = np.array(X)
    return np.mean(X, axis = 0)

def interpolated_average(X: Optional[np.array]) -> np.array:
    '''
    Computes the arithmetic barycenter of a list of arrays after 
    interpolating them to the same length.
    
    Parameters:
        X: array-like, shape = (n_instances, length)
    Returns:
        interpolated_average: np.array, shape = (n_instances, length)
    '''
    if type(X) != np.array:
        X = np.array(X, dtype = object)
    length = np.mean(np.array(list(map(len, X))), dtype = int)
    interpolated_candidates = [interpolate(arr, length) for arr in X]
    return average_barycenter(interpolated_candidates)

################
# Scoping Maps #
################

metrics  = {'euclidean' : euclidean_distance,
            'correlation' : cross_correlation,
            'dtw' : dynamic_time_warping,
            }

barycenters = {'interpolated' : interpolated_average,
               'average' : average_barycenter
               }

manipulations = {'interpolate' : interpolate,
                 'reinterpolate' : reinterpolate,
                 'pad' : pad,
                 'moving_average' : center_moving_average,
                 }