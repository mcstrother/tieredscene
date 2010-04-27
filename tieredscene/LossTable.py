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
        self._trace = numpy.zeros((State.count_states(image_array, self._label_set), image_array.shape[1]  ) ) # the number in each cell corresponds to the optimal state number for the preceding column
        
        for column in xrange(1, self._table.shape[1]):
            for this_label in self._label_set.all_labels:
                for previous_label in self._label_set.all_labels:
                    for relative_positioning in xrange(6):
                        pass
        
    def _U(self, state, column):
        """Returns the sum of the vertical smoothness loss and the data smoothness loss the given state/column pair
        
        This sum is significant because it can be
        calculated without considering the state
        of any column other than the one specified.
        """
        return self._vslc.get_loss(state, column) + self._dlc.get_loss(state, column)
        
    
    def _get_decoupled_functions(self, hslc, col, previous_label, this_label, positioning ):  #TODO: this is an awful method name
        """The horizontal loss between `column` and the previous column broken down into functions of i, j, ib, and jb
        
        A key observation of the Felzenszwalb paper is that if we fix the state of
        the current column (label, i, and b), the label of the previous column, 
        and the relative position of
        i,j, ib, and jb (see below), the HorizontalSmoothnessLoss between a given
        column and the column before it can be expressed as a sum of functions 5
        functions -- 4 functions of only i, j, ib, or jb, and one constant function.
        (Where i and j refer to the properties of the state assigned to the current
        function, and ib and jb refer to the properties of the state assigned
        to the previous function.)
        
        
        Parameters
        ----------
        hslc :  a `HorizontalSmoothnessLossCache` object
        col : the column of the loss table that we are currently calculating
        previous_label : any label in self._label_set
        this_label : any label in self._label_set
        positioning : an integer representing the relative positing of i,j, i_bar, and j_bar 
        
        Returns
        -------
        (I, J, Ib, Jb, C) : a tuple of functions where the HorizontalSmoothnessLoss is equal to
            I(i) + J(j) + Ib(ib) + Jb(jb) + C()
        
        The mapping for the `positioning` parameter is:
        1. i, j, ib, jb
        2. i, ib, j, jb
        3. i, ib, jb, j
        4. ib, i, j, jb
        5. ib, i, jb, j
        6. ib, jb, i, j
        """
        table = hslc.table
        ln1 = self._label_set.label_to_int(previous_label)
        ln2 = self._label_set.label_to_int(this_label)
        n = self._image_array.shape[0]
        t = self._label_set.label_to_int(self._label_set.top)
        b = self._label_set.label_to_int(self._label_set.bottom)
        
        
        C = lambda x: table[n, col, ln1, ln2] 
        if positioning == 1:
            I = lambda i: table[i, col, t, t] - table[i,col, t, ln2]
            J = lambda j: table[j, col, t, ln2] - table[j, col, t, b]
            Ib = lambda ib: table[ib, col, t, b] - table[ib, col, ln1, b]
            Jb = lambda jb: table[jb, col, ln1, b] - table[jb, col, b, b]
        elif positioning == 2:
            I = lambda i: table[i, col, t,t] - table[i, col, t, ln2]
            Ib = lambda ib: table[ib, col, t, ln2] - table[ib, col, ln1, ln2]
            J = lambda j: table[j, col, ln1, ln2] - table[j, col, ln1, b]
            Jb = lambda jb: table[jb, col, ln1, b] - table[jb, col, b, b]
        elif positioning == 3:
            I = lambda i: table[i, col, t, t] - table[i, col,t, ln2]
            Ib = lambda ib: table[ib, col, t, ln2] - table[ib, col, ln1, ln2]
            Jb = lambda jb: table[jb, col, ln1, ln2] - table[jb, col, b, ln2]
            J = lambda j: table[j, col, b, ln2] - table[j, col, b, b] 
        elif positioning == 4:
            Ib = lambda ib: table[ib, col, t, t] - table[ib, col, ln1, t]
            I = lambda i: table[i, col, ln1, t] - table[i, col, ln1, ln2]
            J = lambda j: table[j, col, ln1, ln2] - table[j, col, ln1, b]
            Jb = lambda jb: table[jb, col, ln1, b] - table[jb, col, b, b]
        elif positioning == 5:
            Ib = lambda ib: table[ib, col, t, t] - table[ib, col, ln1, t]
            I = lambda i: table[i,  col, ln1, t] - table[i, col, ln1, ln2]
            Jb = lambda jb: table[jb, col, ln1, ln2] - table[jb, col,b, ln2]
            J = lambda j: table[j, col, b, ln2] - table[j, col, b, b]
        elif positioning == 6:
            Ib = lambda ib: table[ib, col, t, t] - table[ib, col, ln1, t]
            Jb = lambda jb: table[jb, col, ln1, t] - table[jb, col, b, t]
            I = lambda i: table[i, col, b, t] - table[i, col, b, ln2]
            J = lambda j: table[j, col, b, ln2] - table[j, col, b, b]
        else:
            raise Exception("Invalid positioning.  Expected a number in range(1,6), but got " + str(positioning))
        return (I,J,Ib,Jb,C)
            
        
        