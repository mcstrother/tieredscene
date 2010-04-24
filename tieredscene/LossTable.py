'''
Created on Apr 23, 2010

@author: mcstrother
'''
import numpy
from tieredscene.State import State
from tieredscene.DataLossCache import DataLossCache
from tieredscene.HorizontalSmoothnessLossCache import HorizontalSmoothnessLossCache
from tieredscene.VerticalSmoothnessLossCache import VerticalSmoothnessLossCache

class LossTable(object):
    '''
    Maintains a dynamic programming table of dimension
    (# of possible states, # of columns in the image).
    Once initialized, cell a,b of the table represents
    the loss of the optimal column-state mapping if the
    image consists of only columns 0 through b and
    column b is assign the state corresponding to the
    int a. 
    
     
    '''


    def __init__(self, image_array, data_loss_func, smoothness_loss_func):
        '''
        Constructor
        '''
        if not ( data_loss_func.label_set() == smoothness_loss_func.label_set() ):
            raise ValueError('data_loss_func and smoothness_loss_func must operate over the same label set')
        self._label_set = data_loss_func.label_set
        self._image_array = image_array
        self._dlc = DataLossCache(self._image_array, data_loss_func)
        self._hslc = HorizontalSmoothnessLossCache(self._image_array, smoothness_loss_func)
        self._vslc = VerticalSmoothnessLossCache(self._image_array, smoothness_loss_func)
        #initialize the table and fill in the first column
        self._table = numpy.zeros((State.count_states(image_array, self._label_set), image_array.shape[1]  ) )
        self._table[:, 0] = [self._U(State.from_int(i, self._label_set, self._image_array), 0) for i in range(self._table.shape[0] ) ] #init the first column of the table
        
        
        
    def _U(self, state, column):
        """Returns the sum of the vertical smoothness loss and the data smoothness loss the given state/column pair
        
        This sum is significant because it can be
        calculated without considering the state
        of any column other than the one specified.
        """
        return self._vslc.get_loss(state, column) + self._dlc.get_loss(state, column)
        
        
        
        