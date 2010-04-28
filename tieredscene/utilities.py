'''
Created on Apr 27, 2010

@author: mcstrother
'''

import numpy

def running_min(arr, reverse = False):
    """
    Parameters
    ----------
    arr : a 1d numpy array
    
    Returns
    -------
    min_arr : the running min of the input
    min_ind_arr: the running argmin of the input
    """
    if reverse:
        arr = arr[::-1]
    min_arr = numpy.empty(arr.shape)
    min_ind_arr = numpy.empty(arr.shape) 
    min_arr[0] = arr[0]
    min_ind_arr[0] = 0
    for i in xrange(1, arr.shape[0]):
        if arr[i] < min_arr[i-1]:
            min_arr[i] = arr[i]
            min_ind_arr[i] = i
        else:
            min_arr[i] =  min_arr[i-1]
            min_ind_arr[i] = min_ind_arr[i-1]
    if reverse:
        arr = arr[::-1]
        min_ind_arr = len(min_ind_arr)-1-min_ind_arr
    return (min_arr, min_ind_arr)
    