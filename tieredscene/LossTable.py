'''
Created on Apr 23, 2010

@author: mcstrother
'''
import numpy
from tieredscene.State import State
from tieredscene.DataLossCache import DataLossCache
from tieredscene.HorizontalSmoothnessLossCache import HorizontalSmoothnessLossCache
from tieredscene.VerticalSmoothnessLossCache import VerticalSmoothnessLossCache
from tieredscene.utilities import running_min
import logging

log = logging.getLogger('tieredscene.LossTable')
flog = logging.getLogger('tieredscene.LossTable._FTables')

class _UTable(object):
    """
    Precalculates and stores the part of the loss that can be calculated
    on a single-column basis.
    """
    
    def __init__(self, image_array, data_loss_func, smoothness_loss_func):
        log.info('Precalculating Vertical Smoothness Loss and Data Loss')
        self._vslc = VerticalSmoothnessLossCache(image_array, smoothness_loss_func)
        self._dlc = DataLossCache(image_array, data_loss_func)
        log.info('done precalculating vertical smoothness loss and data loss')
    
    def get_loss(self, state, column):
        """O(1) calculation of sum of data loss and vertical smoothness loss for state/column pair
        """
        return self._vslc.get_loss(state, column) + self._dlc.get_loss(state, column)



class _FTables(object):
    
    def __init__(self, hslc, label_set, image_array, loss_table_column, col_num, previous_label, this_label ):
        n = image_array.shape[0]
        self._fs = [None] *6
        for rel_pos in xrange(6):
            I, J, Ib, Jb, C = hslc.get_decoupled_functions(col_num,previous_label, this_label, rel_pos) 
            Ep = self._get_E_prime(n, label_set, image_array, loss_table_column,  col_num, previous_label, Ib, Jb, rel_pos)
            h = self._get_h(n, label_set, image_array, loss_table_column,  col_num, previous_label, Ep, Ib, Jb, rel_pos)
            self._fs[rel_pos] = lambda i,j : (I(i) + J(j) + C() + h(i,j)[0], 
                                              h(i,j)[1], 
                                              h(i,j)[2])
        self._label_set = label_set
        self._previous_label = previous_label
        self._image_array = image_array
        
    def get_best_prev_state(self, i, j, rel_pos = None):
        if not rel_pos is None:
            return self._fs[rel_pos](i,j)
        
        best_prev_pair= None
        best_prev_state_value = None
        for rel_pos in xrange(6):
            f_out = self.get_best_prev_state(i, j, rel_pos)
            curr_value = f_out[0]
            if best_prev_pair is None or curr_value < best_prev_state_value:
                ib = f_out[1]
                jb = f_out[2]
                best_prev_pair = (ib,jb)
                best_prev_state_value = curr_value
        ib, jb = best_prev_pair
        best_prev_state = State(ib, jb, self._previous_label, self._label_set, self._image_array)
        return (best_prev_state,  curr_value)

    def _get_E_prime(self, n, label_set, image_array, loss_table_column,  col, previous_label, Ib, Jb, rel_pos):
        """Caclulated E' in O(n^2) time as described in equation 19 of the Felzenszwalb paper
        
        Parameters
        ----------
        `n` :
        label_set :
        image_array :
        loss_table_column :
        col :
        previous_label :
        Ib : 
        Jb :
        
        Returns
        -------
        ep_list : a list of all 6 E' functions, one corresponding to each
            possible relative positioning.  e.g. the one corresponding
            to the relative positioning 0 is ep_list[0].  Each E' function
            takes a different list of inputs, but each returns a value and
            and index
        
        """
        ep_list = [None]*6
        #get ep for all relative positioninings except case 1
        ep_table = [None] * n
        ep_ind_table = [None] * n
        reverse_ep_table = [None] * n
        reverse_ep_ind_table = [None] * n
        for jb in xrange(n):
            val_arr = numpy.empty(jb+1)
            for ib in xrange(jb+1):
                previous_state = State(ib, jb, previous_label, label_set, image_array)
                val_arr[ib]= loss_table_column[previous_state.as_int()] + Ib(ib)
            ep_table[jb], ep_ind_table[jb] = running_min(val_arr)
            #ep_table[jb][x] = for a given jb, what is the minimum value of loss_table_column[previous_state.as_int()] + Ib(ib) if ib is allowed to run from 0 to x
            #ep_ind_table[jb][x] gives the value  of ib for which the min above occurs
            #the reverse_ep... tables work the same, except the min is over i running from x to jb
            reverse_ep_table[jb], reverse_ep_ind_table[jb] = running_min(val_arr, reverse = True)
        ep_list[5] = lambda jb: (ep_table[jb][jb], ep_ind_table[jb][jb] )
        ep_list[2] =  lambda i, jb: (reverse_ep_table[jb][i], reverse_ep_ind_table[jb][i]  )
        ep_list[4] = ep_list[3] = lambda i, jb: (ep_table[jb][i], ep_ind_table[jb][i] )
        ep_list[0]= lambda j, jb: (reverse_ep_table[jb][j], reverse_ep_ind_table[jb][j])
        #get ep for relative_positioning = 1
        ep_list[1] = self._rel_pos_1_ep_subcase(n, label_set, image_array, loss_table_column, col, previous_label, Ib, Jb, rel_pos)
        return ep_list[rel_pos]
    
    def _rel_pos_1_ep_subcase(self, n, label_set, image_array, loss_table_column,  col, previous_label, Ib, Jb, rel_pos):
        if rel_pos == 1:
            ep_table2 = [None] * n
            ep_ind_table2 = [None] * n
            for ib in xrange(n):
                val_arr = numpy.empty(n-ib)
                for jb in xrange(ib, n):
                    previous_state = State(ib,jb, previous_label, label_set, image_array)
                    val_arr[jb-ib] = loss_table_column[previous_state.as_int()] + Jb(jb)
                ep_table2[ib], ep_ind_table2[ib] = running_min(val_arr, reverse = True)
                ep_ind_table2[ib] = ep_ind_table2[ib] + ib
            return lambda j, ib: (ep_table2[ib][j-ib], ep_ind_table2[ib][j-ib]) #note that this returns a value and an index for __jb__, not ib
        else:
            return None
        
    
    def _get_h(self, n, label_set, image_array, loss_table_column,  col, previous_label, Ep, Ib, Jb, rel_pos):
        """Calculate h as described in equation 22 of the Felzenszwalb paper
        
        Runs in O(n^2) time.
        
        Parameters
        ----------
        
        Returns
        -------
        """
        h_list = [None]*6
        
        h_table = [None] * n
        h_ind_table = [None]*n
        reverse_h_table = [None] * n
        reverse_h_ind_table = [None]* n
        ep_ind_table = [None]*n
        
        if rel_pos == 0:
            for j in xrange(n):
                val_arr = numpy.empty(n-j)
                ep_ind_arr = numpy.empty(n-j)
                for jb in xrange(j,n):
                    val_arr[jb-j] = Jb(jb) + Ep(j,jb)[0]
                    ep_ind_arr[jb-j] = Ep(j,jb)[1]
                reverse_h_table[j], reverse_h_ind_table[j] = running_min(val_arr, reverse = True)
                ep_ind_table[j] = numpy.array([ep_ind_arr[ind] for ind in reverse_h_ind_table[j] ])
                reverse_h_ind_table[j] = reverse_h_ind_table[j] + j #represents best jb
            h_out = lambda i,j: (reverse_h_table[j][0], ep_ind_table[j][0], reverse_h_ind_table[j][0] )
        elif rel_pos == 1:
            for j in xrange(n):
                val_arr = numpy.empty(j+1)
                ep_ind_arr = numpy.empty(j+1)
                for ib in xrange(j+1):
                    val_arr[ib] = Ib(ib) + Ep(j, ib)[0]
                    ep_ind_arr[ib] = Ep(j,ib)[1]
                reverse_h_table[j], reverse_h_ind_table[j] = running_min(val_arr, reverse = True)
                ep_ind_table[j] = numpy.array([ep_ind_arr[ind] for ind in reverse_h_ind_table[j] ])
            h_out = lambda i,j: (reverse_h_table[j][i], reverse_h_ind_table[j][i], ep_ind_table[j][i])
        elif rel_pos == 2:
            for i in xrange(n):
                val_arr = numpy.empty(n-i)
                ep_ind_arr = numpy.empty(n-i)
                for jb in xrange(i, n):
                    val_arr[jb-i] = Jb(jb) + Ep(i,jb)[0]
                    ep_ind_arr[jb-i] = Ep(i,jb)[1]
                h_table[i], h_ind_table[i] = running_min(val_arr)
                ep_ind_table[i] = numpy.array([ep_ind_arr[int(ind)] for ind in h_ind_table[i] ]) #TODO: problem here due to numpy 0-d array
                h_ind_table[i] = h_ind_table[i] + i #represents best jb
            h_out = lambda i,j: (h_table[i][j-i], ep_ind_table[i][j-i], h_ind_table[i][j-i])
        elif rel_pos == 3:
            for i in xrange(n):
                val_arr = numpy.empty(n-i)
                ep_ind_arr = numpy.empty(n-i)
                for jb in xrange(i,n):
                    val_arr[jb-i] = Jb(jb) + Ep(i,jb)[0]
                    ep_ind_arr[jb-i] = Ep(i,jb)[1]
                reverse_h_table[i], reverse_h_ind_table[i] = running_min(val_arr, reverse = True)
                ep_ind_table[i] = numpy.array([ep_ind_arr[ind] for ind in reverse_h_ind_table[i] ])
                reverse_h_ind_table[i] = reverse_h_ind_table[i] + i # best jb
            h_out = lambda i,j: (reverse_h_table[i][j], ep_ind_table[i][j], reverse_h_ind_table[i][j] )
        elif rel_pos == 4:
            for i in xrange(n):
                val_arr = numpy.empty(n-i)
                ep_ind_arr = numpy.empty(n-i)
                for jb in xrange(i,n):
                    val_arr[jb-i] = Jb(jb) + Ep(i,jb)[0]
                    ep_ind_arr[jb-i] = Ep(i,jb)[1]
                h_table[i], h_ind_table[i] = running_min(val_arr)
                ep_ind_table[i] = numpy.array([ep_ind_arr[ind] for ind in h_ind_table[i] ])
                h_ind_table[i] = h_ind_table[i] + i
            h_out = lambda i,j: (h_table[i][j], ep_ind_table[i][j], h_ind_table[i][j] ) #(value, ib, jb)
        elif rel_pos == 5:
            val_arr = numpy.empty(n)
            ep_ind_arr = numpy.empty(n)
            for jb in xrange(n):
                val_arr[jb] = Jb(jb) + Ep(jb)[0]
                ep_ind_arr[jb] = Ep(jb)[1]
            h_table[0], h_ind_table[0] = running_min(val_arr)
            ep_ind_table[0] = numpy.array([ep_ind_arr[ind] for ind in h_ind_table[0] ]) 
            h_out = lambda i,j : (h_table[0][i], ep_ind_table[0][i], h_ind_table[0][i])
        
        return h_out
                
        
    

import functools

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
        if not ( data_loss_func.label_set == smoothness_loss_func.label_set ):
            raise ValueError('data_loss_func and smoothness_loss_func must operate over the same label set')
        
        label_set = data_loss_func.label_set
        log.info('Precalculating horizontal smoothness loss')
        hslc = HorizontalSmoothnessLossCache(image_array, smoothness_loss_func)
        log.info('horizontal smoothness loss precalculated')
        u = _UTable(image_array, data_loss_func, smoothness_loss_func)
        #initialize the table and fill in the first column
        self._table = numpy.zeros((State.count_states(image_array, label_set), image_array.shape[1]  ) ) #number of states by number of columns
        self._table[:, 0] = [u.get_loss(State.from_int(i, label_set, image_array), 0) + hslc.get_loss(None, State.from_int(i,label_set, image_array), 0)
                              for i in range(self._table.shape[0] ) ] #init the first column of the table
        self._trace = numpy.zeros((State.count_states(image_array, label_set), image_array.shape[1]  ) ) # the number in each cell corresponds to the optimal state number for the preceding column
        n = image_array.shape[0]
        
        self._label_set = label_set
        self._image_array = image_array
        self._u = u
        self._hslc = hslc
        
        
        for column in xrange(1, self._table.shape[1]):
            log.info('Processing column ' + str(column))
            for this_label in label_set.middle:
                for previous_label in label_set.middle:
                    ftables = _FTables(hslc, label_set, image_array, self._table[:,column-1], column, previous_label, this_label)
                    for i in xrange(n+1):
                        for j in xrange(i, n+1):
                            curr_state = State(i, j, this_label, label_set, image_array)
                            #best_prev_state, best_prev_value = ftables.get_best_prev_state(i, j)
                            best_prev_state, best_prev_value = self._brute_get_best_previous(curr_state, column)
                            self._table[curr_state.as_int(), column] = best_prev_value + u.get_loss(curr_state, column)
                            self._trace[curr_state.as_int(), column] = best_prev_state.as_int()
        


    def get_optimal_state_list(self):
        m = self._image_array.shape[1] #m = number of columns in the image
        state_ind_list = [None] * m
        state_ind_list[m-1] = numpy.argmin(self._table[:,m-1]) #the best final state
        for col in xrange(m-2, -1, -1):
            state_ind_list[col] = self._trace[ state_ind_list[col+1]  ,col+1]
            
        state_list = [State.from_int(ind, self._label_set, self._image_array) for ind in state_ind_list]
        return state_list 
    
    def _brute_get_best_previous(self, current_state, column):
        """
        Gets the best previous state based on the current state
        and current column, but does so in O(n^2) computations
        instead of O(1).  To do this in O(1), use _Ftables 
        """
        num_states = State.count_states(self._image_array, self._label_set)
        n = self._image_array.shape[0]+1
        best_prev_state = None
        best_prev_value = None
        for sn in xrange(num_states):
            prev_state = State.from_int(sn, self._label_set, self._image_array)
            value = self._hslc.get_loss(prev_state, current_state, column ) + self._table[prev_state.as_int(), column-1]
            if best_prev_value is None or best_prev_value > value:
                best_prev_state = prev_state
                best_prev_value = value
        return (best_prev_state, best_prev_value)
        
    
    """

    def get_optimal_state_list(self):
        state_inds = numpy.argmin(self._table, axis =0).tolist()
        return [State.from_int(ind, self._label_set, self._image_array) for ind in state_inds  ]     
    """