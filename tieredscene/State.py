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
        i : the top-middle boundary (pixel i is included in middle)
        j : the middle-bottom boundary (pixel j is included in bottom)
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
        if (not int(i) == i) or (not int(j) == j):
            raise ValueError('i and j must have integer values')
        if not ((0 <= i <= image_array.shape[0]) and (i <= j <= image_array.shape[0])):
            raise ValueError("i,j values (" + str(i) + ', ' + str(j) + ") are not valid for image_array of shape "+ str(image_array.shape))
        
    
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
        return self._label_set.label_to_int(self.mlabel)
    @property
    def tee(self):
        return self._label_set.label_to_int(self._label_set.top)
    @property
    def bee(self):
        return self._label_set.label_to_int(self._label_set.bottom)
    @property
    def label_set(self):
        return self._label_set
    
    @classmethod
    def from_int(cls, ind, label_set, image_array ):
        """Creates and returns a state object corresponding to the given int
        
        """
        rowc = float(image_array.shape[0])+1
        
        el_multiplier = (rowc/2) * (rowc+1)
        el_num = int(math.floor(ind/( el_multiplier )))
        ind -= el_multiplier * el_num
        
        a = -.5
        b = (rowc + .5)
        c = -ind 
        i = math.floor(( -b + math.sqrt(math.pow(b,2) - 4*a*c) )/(2*a) )
        ind = ind - i * (-.5 * i + rowc + .5 )
        
        j = ind + i
        
        return State(i, j, label_set.middle[el_num], label_set, image_array)
    
    
    def as_int(self):
        """Returns an int that is unique to this State.
        
        The returned int lies in the range 0 to (# of rows in image) + 
        (# columns in image) + (# of middle labels) - 1
        """
        r = self._image_array_shape[0]+1 # number of rows in the image
        i_term = self.i * (-.5 * self.i  + r + .5 )
        
        j_term = self.j - self.i
        
        el_term = self.el * r * .5 * (r+1)
        
        return (i_term + j_term + el_term)
        
    @classmethod
    def count_states(cls, image_array, label_set):
        """Return the number of valid states  for this image_arry and label_set
        
        Since a "state" is defined by i, j, and mlabel,
        the total number of valid states is equal to the
        total number of labels (including the top and the bottom)
        times the total number of valid i,j pairs.
        """
        el = len(label_set.middle)
        r = image_array.shape[0] + 1
        out = el * r/2.0*(r+1)
        if int(out) != out:
            raise Error('An internal error has occured.')
        return int(out)
    
    def to_array(self):
        """Get a 1d numpy array filled with numbers corresponding to the state.
        
        For example, if the state has an image_array of shape[0] = 10,
        and i and j are 3 and  7 respectively, and el is 2, and the numbers
        corresponding to the top and bottom labels are 3 and 4 respectively,
        the output would be the array
        [3 3 3 2 2 2 2 4 4 4]
        """
        num_rows = self._image_array_shape[0] # number of rows in the image
        out = numpy.empty(num_rows)
        out[:self.i] = self.tee
        out[self.i:self.j] = self.el
        out[self.j:num_rows] = self.bee
        return out
    
    def get_row_label(self, row):
        """Given a row number, return the label of the pixel in that row
        
        Parameters
        ----------
        row : the row number of the pixel of interest
        
        Returns
        -------
        label : the label of the pixel in that row
        """
        if 0 <= row < self.i:
            return self._label_set.top
        elif self.i <=row < self.j:
            return self.mlabel
        elif self.j <=row < self._image_array_shape[0]:
            return self._label_set.bottom
        else:
            raise IndexError('Invalid row number, '  + str(row))
    
    def __str__(self):
        return '(' + str(self.i) +',' + str(self.j) + ','+str(self.mlabel) + ')'
    
    
    @classmethod
    def get_relative_positioning(cls, state1, state2):
        """Get the number of the relative positioning of two states
        
        
        
        ib and jb refer to state 1
        i and j refer to state 2
        The number returned is dependent on the ordering of
        these 4 numbers.  The mapping is as follows
        
        0. i, j, ib, jb
        1. i, ib, j, jb
        2. i, ib, jb, j
        3. ib, i, j, jb
        4. ib, i, jb, j
        5. ib, jb, i, j
        
        Parameters
        ----------
        state1 :
        state2 :
        
        Returns
        -------
        rel_pos : a number between 0 and 5 inclusive
        """
        ib = state1.i
        jb = state1.j
        i = state2.i
        j = state2.j
        
        #This could probably be slightly more efficient, but
        #not much more
        if i <= j <= ib <= jb:
            return 0
        elif i <= ib <= j <= jb:
            return 1
        elif i <= ib <= jb<= j:
            return 2
        elif ib <= i <= j <= jb:
            return 3
        elif ib <= i <= jb <=j:
            return 4
        elif ib <= jb <= i <= j:
            return 5
        else:
            raise ValueError('Positioning of states does not match any valid relative positioning')
        