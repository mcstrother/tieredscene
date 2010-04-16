'''
Created on Apr 4, 2010

@author: mcstrother
'''


class DataLossFunction(object):
    """
    An object representing the data loss function
    being used for a run of the tiered scene labeling
    algorithm.
    
    A data loss function can be any function which, given
    a pixel in an image and a label assigned to it, returns
    a numerical loss which is incurred by assigning that
    label to that pixel.  The data loss function is only
    allowed to take into account the pixel's x-y coordinates
    in the image and the pixel's intensity.  In contrast, 
    a smoothness loss function may take into account the 
    intensities and labels of neighboring pixels.
    """
    _label_set = None
    
    def __init__(self):
        if self._label_set is None:
            raise NotImplementedError("Subclasses of DataLossFunction must define a '_label_set' class variable.")
    
    
    
    def __call__(self, pixel, label):
        """
        """
        raise NotImplementedError("Subclasses of DataLossFunction must override __call__")


