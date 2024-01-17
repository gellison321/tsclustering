import numpy as np
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
    ret = np.cumsum(array)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

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
    length = np.mean(np.array(list(map(len, X))), dtype = int)
    interpolated_candidates = [interpolate(arr, length) for arr in X]
    return average_barycenter(interpolated_candidates)

################
# Scoping Maps #
################

utils = {'interpolate' : interpolate,
         'reinterpolate' : reinterpolate,
         'pad' : pad,
         'moving_average' : center_moving_average,
         'interpolated_barycenter' : interpolated_average,
         'average_barycenter' : average_barycenter
        }