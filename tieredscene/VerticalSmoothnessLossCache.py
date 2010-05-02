'''
Created on Apr 16, 2010

@author: mcstrother
'''

import numpy
import Pixel
import logging
import math

log = logging.getLogger('tieredscene.VerticalSmoothnessLossCache')


class VerticalSmoothnessLossCache(object):
    '''
    An object which, once initialized, allows constant 
    time calculation of the vertical component of the
    smoothness loss  implied by a 
    particular state at a particular column.
    '''


    def __init__(self, image_array, loss_function):
        '''
        Parameters
        ----------
        image_array : a 2d numpy ndarray representing the image
            being processed
        loss_function : A SmoothnessLossFunction object which defines
            the smoothness loss function that is to be used
            
        Note that the vertical loss for a pixel in the top row of
        the image is calculated as 
        loss_function.vertical_loss(None, None, pixel, label).
        
        Note that this function runs in O(n*m*K) time,
        where n = # of rows in the image, m = # of columns
        in the image, and K = the number of labels
        '''
        #precompute a "loss-image" for each label
        loss_images = numpy.empty((image_array.shape[0], 
                                        image_array.shape[1], 
                                        loss_function.label_set.get_label_count() ))
        #calculate the top row of each image
        for label_num, label in enumerate(loss_function.label_set.all_labels):
            for col in xrange(image_array.shape[1]):
                pixel = Pixel.Pixel(image_array, col, 0)
                loss_images[0, col, label_num] = loss_function.vertical_loss(None, None, pixel, label)
        #fill in the rest of the loss images
        for label_num, label in enumerate(loss_function.label_set.all_labels):
            for col in xrange(image_array.shape[1]):
                for row in xrange(1, image_array.shape[0]):
                    pixel1 = Pixel.Pixel(image_array, col, row-1)
                    pixel2 = Pixel.Pixel(image_array, col, row)
                    loss_images[row, col, label_num] = loss_function.vertical_loss(pixel1, label, pixel2, label)
        #compute running sums over all columns of each loss-image
        self._integral = numpy.empty((image_array.shape[0], 
                                      image_array.shape[1], 
                                      loss_function.label_set.get_label_count()))
        for label_num, label in enumerate(loss_function.label_set.all_labels):
            self._integral[:,:,label_num] = numpy.cumsum( loss_images[:,:,label_num], axis=0 )
        #self._integral is now a 3 dimensional array.  The number in self.integral[rw, col, ln]
        # represents the cumulative vertical smoothness loss if all pixels in column col from row 0
        # through row rw are assigned the label number ln
        
        #save references for later.  needed in self.get_loss
        self._loss_function = loss_function
        self._image_array = image_array
        
    def get_loss(self, state, column):
        """Returns the total vertical smoothness loss implied
        by assigning a given state to a given column.
        
        Note that this method always terminates
        in constant time.
        
        Parameters
        ----------
        state : a state object
        column: a column number
        
        Returns
        -------
        loss : the vertical smoothness loss implied by 
            assigning state to column
        """
        ls = state.label_set
        loss = 0
        if state.i >0: #if state.i==0 then no pixels are labeled as "top"
            loss += self._integral[state.i-1, column, state.tee] #top loss
            if state.j > state.i:
                #transition from top to middle
                pixel1 = Pixel.Pixel(self._image_array, column, state.i-1)
                pixel2 = Pixel.Pixel(self._image_array, column, state.i)
                loss += self._loss_function.vertical_loss(pixel1, ls.top, pixel2, state.mlabel)
        if state.j > state.i:
            #middle loss
            loss += self._integral[state.j-1, column, state.el] - self._integral[state.i, column, state.el]
            if self._image_array.shape[0] > state.j:
                #transition from middle to bottom
                pixel1 = Pixel.Pixel(self._image_array, column, state.j-1)
                pixel2 = Pixel.Pixel(self._image_array, column, state.j)
                loss += self._loss_function.vertical_loss(pixel1, state.mlabel, pixel2, ls.bottom)
        if self._image_array.shape[0] > state.j:
            #bottom loss
            loss += self._integral[-1, column, ls.label_to_int(ls.bottom) ] - self._integral[state.j, column, ls.label_to_int(ls.bottom) ]
        if math.isnan(loss) or math.isinf(loss):
            log.error('Internal Error: get_loss has returned an invalid loss, ' + str(loss))
        return loss
        
        
        