'''
Created on Apr 4, 2010

@author: mcstrother
'''

import numpy
import math


class State(object):
    
    def __init__(self, i, j, mlabel, label_set, image_array):
        """Creates a State object.
        
        Paramters:
        ---------
        i : the top-middle boundary
        j : the middle-bottom boundary
        mlabel : the label assigned to the middle region
        label_set : the label set from which mlabel is drawn
        image : a 2d array representing the image on which the DP algorithm is being run
        """
        
        self._label_set = label_set
        self._i = i # T-M boundary
        self._j = j# M-B boundary
        self._mlabel = mlabel # the label of the middle region
        self._image_array_shape = image_array.shape
        
        if not (mlabel in self._label_set.middle):
            raise ValueError("mlabel " + str(mlabel) + " not found in the middle label_set")
        if not (0 <= i <= image_array.shape[0]) and (i <= j <= image_array.shape[0]):
            raise ValueError("i,j values (" + str(i) + ', ' + str(j) + ") are not valid for image_array of shape "+ image_array.shape)
        
    
    @property
    def i(self):
        return self._i
    
    @property
    def j(self):
        return self._j
    
    @property
    def mlabel(self):
        return self._mlabel 
    
    @property
    def el(self):
        return self._label_set.to_int(self.mlabel)
    
    @classmethod
    def from_int(ind, label_set, image_array ):
        """Creates and returns a state object corresponding to the given int
        
        """
        rowc = float(image_array.shape[0])
        
        el_multiplier = (rowc/2) * (rowc+1)
        el_num = math.floor(ind/( el_multiplier ))
        ind = el_multiplier * el_num
        
        
        a = -.5
        b = (rowc - .5)
        c = (rowc - ind )
        i = math.floor(( -b + math.sqrt(math.pow(b,2) - 4*a*c) )/(2*a)   )
        
        #j = ind - i
        
        return State(i, j, label_set.middle[el_num], label_set, image_array)
    
    
    def as_int(self):
        """Returns an int that is unique to this State.
        
        The returned int lies in the range 0 to (# of rows in image) + 
        (# columns in image) + (# of middle labels) - 1
        """
        r = self._image_array_shape[0] # number of rows in the image
        i_term = self.i * (-.5 * self.i  + r + .5 )
        
        j_term = self.j - self.i
        
        el_term = self.el * r * .5 * (r+1)
        
        return (i_term + j_term + el_term)
        
        
        
        