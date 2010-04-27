'''
Created on Apr 17, 2010

@author: mcstrother
'''

import numpy
from tieredscene import Pixel

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
                for row in xrange(1, image_array.shape[0]):
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
        
        
    @property
    def table(self):
        """
        table[row, column, labelnumber1, labelnumber2] = the horizontal smoothness loss incurred by labeling rows 0
            through `row` in column-1 labelnumber1 and labeling rows 0 through `row` in `column` labelnumber2
        """
        return self._integral
        