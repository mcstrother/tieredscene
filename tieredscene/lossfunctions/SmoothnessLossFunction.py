'''
Created on Apr 4, 2010

@author: mcstrother
'''


class SmoothnessLossFunction(object):
    _label_set = None

    
    def __call__(self, pixel1, label1, pixel2, label2):
        """Gets the smoothness loss implied by assigning labels to pixels.
        
        pixel1 must be directly above or to the left of pixel2
        If you know the spatial relationship between pixel1 and
        pixel2, you should just use vertical_loss or horizontal_loss
        accordingly, as this is slightly less efficient
        """
        if (pixel1.column == pixel2.column-1) and pixel1.row == pixel2.row:
            return self.horizontal_loss(pixel1, label1, pixel2, label2)
        elif (pixel1.row == pixel2.row-1) and pixel1.column == pixel2.column:
            return self.vertical_loss(pixel1, label1, pixel2, label2)
        else:
            raise ValueError('pixel1 must be either directly to the left or directly above pixel2')
            
    def vertical_loss(self, pixel1, label1, pixel2, label2):
        """Given pixel1 directly above pixel2, calculates the
        vertical loss implied by assigning label1 to pixel1 and
        label2 to pixel2.
        
        Subclasses of SmoothnessLossFunction must override
        this function and must support a call to
        vertical_loss(None, None, pixel, label), in order to calculate
        loss for the top row of a given image.
        """
        raise NotImplementedError()
    
    def horizontal_loss(self, pixel1, label1, pixel2, label2):
        """Given pixel1 directly to the left of pixel2, calculates
        the horizontal loss implied by assigning label1 to pixel1
        and label2 to pixel2.
        
        Subclasses of SmoothnessLossFunction must override
        this function and must support a call to
        horizontal_loss(None, None, pixel, label) in order
        to calculate the loss for the leftmost row of a given
        image.
        """
        raise NotImplementedError()
    
    @property
    def label_set(self):
        if self._label_set:
            raise ValueError("Subclass of SmoothnessLossFunction has not defined a _label_set variable.")
        else:
            return self._label_set