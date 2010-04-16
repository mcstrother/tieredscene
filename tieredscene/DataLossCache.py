'''
Created on Apr 15, 2010

@author: mcstrother
'''

import numpy
import Pixel

class DataLossCache(object):
    '''
    An object which, once initialized, allows constant 
    time calculation of the data loss implied by a 
    particular state at a particular column.
    
    '''


    def __init__(self, image_array, loss_function ):
        '''
        Parameters
        ----------
        image_array : a 2d numpy ndarray representing the image
            being processed
        loss_function : A DataLossFunction object which defines
            the data loss function that is to be used
        '''
        #precompute a "loss-image" for each label
        loss_images = numpy.empty((image_array.shape[0], 
                                  image_array.shape[1], 
                                  loss_function.label_set.get_label_count()))
        
        for label_num, label in enumerate(loss_function.label_set.all_labels):
            for col in xrange(image_array.shape[1]):
                for row in xrange(image_array.shape[0]):
                    pixel = Pixel.Pixel(image_array, col, row)
                    loss_images[row, col, label_num] = loss_function(pixel, label)
        #compute running sums over all columns of each loss-image
        self._integral = numpy.empty((image_array.shape[0], 
                                      image_array.shape[1], 
                                      loss_function.label_set.get_label_count()))
        for label_num, label in enumerate(loss_function.label_set.all_labels):
            self._integral[:,:,label_num] = numpy.cumsum( loss_images[:,:,label_num], axis=0 )
        
    def get_data_loss(self, state, column):
        """
        Parameters
        ----------
        state : a state object
        column: a column number
        
        Returns
        -------
        loss : the loss implied by assigning
        """
        ls = state.label_set
        loss =  self._integral[state.i-1 , column, ls.label_to_int(ls.top) ] #top loss
        loss += self._integral[state.j-1, column, state.el] - self._integral[state.i-1, column, state.el] # middle loss 
        loss += self._integral[state.j, column, ls.label_to_int(ls.bottom)] - self._integral[state.j-1, column, ls.label_to_int(ls.bottom)] # bottom loss
        return loss
        