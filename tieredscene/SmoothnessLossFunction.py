'''
Created on Apr 4, 2010

@author: mcstrother
'''


class SmoothnessLossFunction(object):
    _label_set = None
    
    def __init__(self):
        if self._label_set is None:
            raise NotImplementedError("Subclasses of SmoothnessLossFunction must define a '_label_set' class variable.")
    
    def __call__(self, pixel1, label1, pixel2, label2):
        """
        """
        if (pixel1.column == pixel2.column-1) and pixel1.row == pixel2.row:
            self.horizontal_loss(pixel1, label1, pixel2, label2)
        elif (pixel1.row == pixel2.row-1) and pixel1.column == pixel2.column:
            self.vertical_loss(pixel1, label1, pixel2, label2)
        else:
            raise ValueError('pixel1 must be either directly to the left or directly above pixel2')
            
    def vertical_loss(self, pixel1, label1, pixel2, label2):
        """Given pixel1 directly above pixel2, calculates the
        vertical loss implied by assigning label1 to pixel1 and
        label2 to pixel2.
        
        Note that subclasses of SmoothnessLossFunction must override
        this function and must support a call to
        vertical_loss(None, None, pixel, label), in order to calculate
        loss for the top row of a given image.
        """
        raise NotImplementedError()
    
    def horizontal_loss(self, pixel1, label1, pixel2, label2):
        raise NotImplementedError()
    
    @property
    def label_set(self):
        return self._label_set