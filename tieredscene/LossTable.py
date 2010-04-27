'''
Created on Apr 23, 2010

@author: mcstrother
'''
import numpy
from tieredscene.State import State
from tieredscene.DataLossCache import DataLossCache
from tieredscene.HorizontalSmoothnessLossCache import HorizontalSmoothnessLossCache
from tieredscene.VerticalSmoothnessLossCache import VerticalSmoothnessLossCache

class _UTable(object):
    
    def __init__(self, image_array, data_loss_func, smoothness_loss_func):
        self._vslc = VerticalSmoothnessLossCache(self._image_array, smoothness_loss_func)
        self._dlc = DataLossCache(self._image_array, data_loss_func)
    
    def get_loss(self, state, column):
        return self._vslc.get_loss(state, column) + self._dlc.get_loss(state, column)



class _FTables(object):
    
    def __init__(self, hslc, label_set, image_array, loss_table_column, col_num, previous_label, this_label ):
        n = image_array.shape[0]
        image_array = image_array
        label_set = label_set
        self._fs = [None] *6
        for relative_positioning in xrange(6):
            I, J, Ib, Jb, C = self._get_decoupled_functions(hslc, n, label_set, col_num, previous_label, this_label, relative_positioning)
            Ep = self._get_E_prime(n, label_set, image_array, loss_table_column, col_num, previous_label, Ib)
            h = self._get_h(n, Ep, Jb)
            self._fs[relative_positioning] = lambda i,j : (I(i) + J(j) + C() + h(i,j)[0], h(i,j)[2], h(i,j)[1])
        
    
    def get_best_i_j(self, i, j, relative_positioning):
        return self._fs[relative_positioning](i,j)

    def _get_E_prime(self, n, label_set, image_array, loss_table_column,  col, previous_label, Ib):
        """Calculate E' in O(n^2) time as described in the Felzenszwalb paper
        
        Runs in O(n^2) time.
        
        Parameters
        ----------
        
        Returns
        -------
        Ep : a function which takes x and jb as parameters and returns a 2-tuple in O(1) time.
            x can be either i, j, or jb, depending on which varialbe immediately follows ib
            in the relative positioning of i, j, ib, and jb. For example, if the relative postioning
            is ib<=i<=jb<=j, x should = i.
            The first value in the tuple is the min over all ib <= i of E[ib, jb] + Ib(ib).
            The second value in the tuple is the value of ib that yields the min.
        """
        ep_table = [None] * n
        ep_ind_table = [None] * n
        for jb in xrange(n):
            min_ind_arr = numpy.empty(jb+1)
            min_val_arr = numpy.empty(jb+1)
            for ib in xrange(jb+1):
                previous_state = State(ib, jb, previous_label, label_set, image_array)
                curr_val = loss_table_column[previous_state.as_int()] + Ib(ib)
                if (ib == 0) or curr_val < min_val_arr[ib-1]:
                    min_val_arr[ib] = curr_val
                    min_ind_arr[ib] = ib
                else:
                    min_val_arr[ib]= min_val_arr[ib-1]
                    min_ind_arr[ib] = min_ind_arr[ib-1]
            ep_table[jb] = min_val_arr
            ep_ind_table[jb] = min_ind_arr
        Ep = lambda i, jb: (ep_table[jb][i], ep_ind_table[jb][i] )
        return Ep
        
    
    def _get_h(self,n, Ep, Jb):
        """Calculate h as described in equation 22 of the Felzenszwalb paper
        
        Runs in O(n^2) time.
        
        Parameters
        ----------
        `Ep` : the Ep function returned by self._get_E_prime
        `Jb` : the Jb function returned by self._get_decoupled functions
        
        Returns
        -------
        `h` : A function which takes two ints i and j and returns a 3-tuple in O(1) time.
            The first value in the tuple is the running min of all jb in [i, j]of Jb(jb)+Ep(i,jb).
            The second value is the value of jb that yields the min.
            The third value is the value of ib that yields the min for the given jb (as returned by the Ep function)
        """
        h_table = [None] * n
        h_ind_table = [None]*n
        ep_ind_table = [None]*n
        for i in xrange(n):
            min_ind_arr = numpy.empty(jb+1)
            min_val_arr = numpy.empty(jb+1)
            ep_min_ind_arr= numpy.empty(jb+1)
            for jb in range(i, n+1):
                curr_val = Jb(jb) + Ep(i, jb)[0]
                if (jb == i) or curr_val < min_val_arr[jb-i-1]:
                    min_val_arr[jb-i] = curr_val
                    min_ind_arr[jb-i] = jb
                    ep_min_ind_arr[jb-i]  = Ep(i,jb)[1]
            h_table[i] = min_val_arr
            h_ind_table = min_ind_arr
        h = lambda i, j: (h_table[i][j-i], h_ind_table[i][j-i], ep_ind_table[i][j-i])
        return h
        
    
    def _get_decoupled_functions(self, hslc, n, label_set, col, previous_label, this_label, positioning ):  #TODO: this is an awful method name
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
        0. i, j, ib, jb
        1. i, ib, j, jb
        2. i, ib, jb, j
        3. ib, i, j, jb
        4. ib, i, jb, j
        5. ib, jb, i, j
        """
        table = hslc.table
        ln1 = label_set.label_to_int(previous_label)
        ln2 = label_set.label_to_int(this_label)
        t = label_set.label_to_int(label_set.top)
        b = label_set.label_to_int(label_set.bottom)
        
        
        C = lambda x: table[n, col, ln1, ln2] 
        if positioning == 0:
            I = lambda i: table[i, col, t, t] - table[i,col, t, ln2]
            J = lambda j: table[j, col, t, ln2] - table[j, col, t, b]
            Ib = lambda ib: table[ib, col, t, b] - table[ib, col, ln1, b]
            Jb = lambda jb: table[jb, col, ln1, b] - table[jb, col, b, b]
        elif positioning == 1:
            I = lambda i: table[i, col, t,t] - table[i, col, t, ln2]
            Ib = lambda ib: table[ib, col, t, ln2] - table[ib, col, ln1, ln2]
            J = lambda j: table[j, col, ln1, ln2] - table[j, col, ln1, b]
            Jb = lambda jb: table[jb, col, ln1, b] - table[jb, col, b, b]
        elif positioning == 2:
            I = lambda i: table[i, col, t, t] - table[i, col,t, ln2]
            Ib = lambda ib: table[ib, col, t, ln2] - table[ib, col, ln1, ln2]
            Jb = lambda jb: table[jb, col, ln1, ln2] - table[jb, col, b, ln2]
            J = lambda j: table[j, col, b, ln2] - table[j, col, b, b] 
        elif positioning == 3:
            Ib = lambda ib: table[ib, col, t, t] - table[ib, col, ln1, t]
            I = lambda i: table[i, col, ln1, t] - table[i, col, ln1, ln2]
            J = lambda j: table[j, col, ln1, ln2] - table[j, col, ln1, b]
            Jb = lambda jb: table[jb, col, ln1, b] - table[jb, col, b, b]
        elif positioning == 4:
            Ib = lambda ib: table[ib, col, t, t] - table[ib, col, ln1, t]
            I = lambda i: table[i,  col, ln1, t] - table[i, col, ln1, ln2]
            Jb = lambda jb: table[jb, col, ln1, ln2] - table[jb, col,b, ln2]
            J = lambda j: table[j, col, b, ln2] - table[j, col, b, b]
        elif positioning == 5:
            Ib = lambda ib: table[ib, col, t, t] - table[ib, col, ln1, t]
            Jb = lambda jb: table[jb, col, ln1, t] - table[jb, col, b, t]
            I = lambda i: table[i, col, b, t] - table[i, col, b, ln2]
            J = lambda j: table[j, col, b, ln2] - table[j, col, b, b]
        else:
            raise Exception("Invalid positioning.  Expected a number in range(6), but got " + str(positioning))
        return (I,J,Ib,Jb,C)


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
        label_set = data_loss_func.label_set
        image_array = image_array
        hslc = HorizontalSmoothnessLossCache(image_array, smoothness_loss_func)
        u = _UTable(image_array, data_loss_func, smoothness_loss_func)
        #initialize the table and fill in the first column
        self._table = numpy.zeros((State.count_states(image_array, label_set), image_array.shape[1]  ) )
        self._table[:, 0] = [u.get_loss(State.from_int(i, label_set, image_array), 0) for i in range(self._table.shape[0] ) ] #init the first column of the table
        self._trace = numpy.zeros((State.count_states(image_array, label_set), image_array.shape[1]  ) ) # the number in each cell corresponds to the optimal state number for the preceding column
        n = image_array.shape[0]
        
        for column in xrange(1, self._table.shape[1]):
            for this_label in label_set.all_labels:
                for previous_label in label_set.all_labels:
                    ftables = _FTables(hslc, label_set, image_array, self._table[:,column-1], column, previous_label, this_label)
                    for i in xrange(n):
                        for j in xrange(i-1, n):
                            curr_state = State(i, j, this_label, label_set, image_array)
                            best_prev_state= None
                            best_prev_state_value = None
                            for rel_pos in xrange(6):
                                f_out = ftables.get_best_i_j(i, j, rel_pos)
                                curr_value = f_out[0]
                                if best_prev_state is None or curr_value < best_prev_state_value:
                                    ib = f_out[1]
                                    jb = f_out[2]
                                    best_prev_state = State(ib, jb, previous_label, label_set, image_array)
                                    best_prev_state_value = curr_value
                            self._table[curr_state.as_int(), column] = best_prev_state_value + u.get_loss(curr_state, column)
                            self._trace[curr_state.as_int(), column] = best_prev_state
        
    

        