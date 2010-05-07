'''
Created on Apr 4, 2010

@author: mcstrother
'''
from tieredscene.Pixel import Pixel

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
    
    
    def brute_get_loss(self, image_array, column, state1, state2):
        """Get the HorizontalSmoothnessLoss between two states.
        
        This is done by brute force.  For an efficient implementation
        of get_loss, see HorizontalSmoothnessLossCache
        
        Parameters
        ----------
        `image_array` :
        `column` : an integer representing the column number of interest
        `state1` : a state object 
        `state2` : a state object
        
        Returns
        -------
        loss :  the horizontal smoothness loss between the two states
        """
        loss = 0
        for row in xrange(image_array.shape[0]):
            if state1 is None:
                p1 = None
                l1 = None
            else:
                p1 = Pixel(image_array, column-1, row)
                l1 = state1.get_row_label(row)
            p2 = Pixel(image_array, column, row)
            l2 = state2.get_row_label(row)
            loss += self.horizontal_loss(p1,l1, p2, l2)
        return loss
        
        
        
    @property
    def label_set(self):
        if self._label_set is None:
            raise ValueError("Subclass of SmoothnessLossFunction has not defined a _label_set variable.")
        else:
            return self._label_set