'''
Created on Apr 17, 2010

@author: mcstrother
'''

import numpy
from tieredscene import Pixel
from tieredscene.State import State

class HorizontalSmoothnessLossCache(object):
    '''
    An object which, once initialized, allows constant 
    time calculation of the horizontal component of the
    smoothness loss implied by assigning a particular
    pair of states to two adjacent columns.
    '''


    def __init__(self, image_array, loss_function):
        '''
        Parameters
        ----------
        image_array : a 2d numpy ndarray representing the image
            being processed
        loss_function : A SmoothnessLossFunction object which defines
            the smoothness loss function that is to be used
            
        Note that the horizontal for a pixel in the leftmost row of
        the image is calculated as 
        loss_function.horizontal_loss(None, None, pixel, label).
        
        Note that this function runs in O(n*m*K^2) time,
        where n = # of rows in the image, m = # of columns
        in the image, and K = the number of labels
        '''
        ls = loss_function.label_set
        #precompute loss images.  one for every pair of labels
        loss_images = numpy.empty((image_array.shape[0],
                                      image_array.shape[1],
                                      ls.get_label_count(),
                                      ls.get_label_count()))
        for ln1, label1 in enumerate(ls.all_labels): #initialize first column of each image
            for ln2, label2 in enumerate(ls.all_labels):
                for row in xrange(image_array.shape[0]):
                    pixel = Pixel.Pixel(image_array, 0, row)
                    loss_images[row, 0, ln1, ln2] = loss_function.horizontal_loss(None, None, pixel, label2)
        for ln1, label1 in enumerate(ls.all_labels): 
            for ln2, label2 in enumerate(ls.all_labels):
                for row in xrange(image_array.shape[0]):
                    for column in xrange(1, image_array.shape[1]):
                        pixel1 = Pixel.Pixel(image_array, column-1, row)
                        pixel2 = Pixel.Pixel(image_array, column, row)
                        loss_images[row, column, ln1, ln2] = loss_function.horizontal_loss(pixel1, label1, pixel2, label2)
        #compute running sums down each column of each loss image
        self._integral = numpy.empty((image_array.shape[0], 
                                      image_array.shape[1], 
                                      ls.get_label_count(),
                                      ls.get_label_count()))
        for ln1, label1 in enumerate(ls.all_labels):
            for ln2, label2 in enumerate(ls.all_labels):
                self._integral[:,:,ln1,ln2] = numpy.cumsum(loss_images[:,:,ln1,ln2], axis=0)
        self._label_set = ls
        
        
    @property
    def table(self):
        """
        table[row, column, labelnumber1, labelnumber2] = the horizontal smoothness loss incurred by labeling rows 0
            through `row` in column-1 labelnumber1 and labeling rows 0 through `row` in `column` labelnumber2
            
        when column = 0, the value is the same for all values of labelnumber1
        """
        return self._integral

    def get_decoupled_functions(self, col, previous_label, this_label, positioning ):
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
        previous_label : any label in self._label_set other than the top or bottom
        this_label : any label in self._label_set other than the top or bottom
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
        table = self.table
        n = self._integral.shape[0]
        ln1 = self._label_set.label_to_int(previous_label)
        ln2 = self._label_set.label_to_int(this_label)
        t = self._label_set.label_to_int(self._label_set.top)
        b = self._label_set.label_to_int(self._label_set.bottom)
    
        #TODO: am I missing a bunch of -1's in here?
        C = lambda : table[n-1, col, b, b] 
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




    def get_loss(self, state1, state2, column2):
        """Get the loss implied by assigning state2 to column2 and state1 to column2-1
        
        Parameters
        ----------
        state1 : a State object
        state2 : a State object
        column2 : an int
        
        Returns
        -------
        loss : a number
        """
        if state1 == None:
            top_loss = self._integral[state2.i-1, 0,0, state2.tee]
            middle_loss = self._integral[state2.j-1, 0,0, state2.el] - self._integral[state2.i-1, 0,0, state2.el]
            bottom_loss = self._integral[-1, 0, 0, state2.bee] - self._integral[state2.j-1, 0,0, state2.bee]
            return top_loss + middle_loss + bottom_loss
            
        ib = state1.i
        jb = state1.j
        i = state2.i
        j = state2.j
        rel_pos = State.get_relative_positioning(state1, state2)
        I,J, Ib, Jb,C = self.get_decoupled_functions(column2, state1.mlabel, state2.mlabel, rel_pos)
        return I(i) + J(j) + Ib(ib) + Jb(jb)  + C()
        