'''
Created on Apr 5, 2010

@author: mcstrother
'''


from tieredscene.lossfunctions import DataLossFunction, SmoothnessLossFunction
from tieredscene import LabelSet
import numpy

gcl_label_set = LabelSet.LabelSet(['L','C','R'], 'T', 'B')


class GCLDataLossFunction(DataLossFunction.DataLossFunction):
    _label_set = gcl_label_set
    
    def __call__(self, pixel, label):
        return 0
        #raise NotImplementedError("TODO: fill this in using the per pixel class confidences from reference 9 from the paper")
    
    

class GCLSmoothnessLossFunction(SmoothnessLossFunction.SmoothnessLossFunction):
    _label_set = gcl_label_set
    _hloss_table = numpy.array([[0, 1, 4, 1, 3 ],
                                [1, 0, 1, 1, 1 ],
                                [4, 1, 0, 1, 1 ],
                                [1, 1, 1, 0, 2 ],
                                [1, 1, 3, 2, 0 ]
                                         ])
    _w_coeff = .1 #TODO: may need to be tweaked
    min_gradient = .1
    
    #TODO: calculate the w term
    def __init__(self, image_array):
        SmoothnessLossFunction.SmoothnessLossFunction.__init__(self)
        if len(image_array.shape) != 2:
            raise ValueError('image_array must have dimension 2.  Check that a black and white image is being used.')
        self._vgrad, self._hgrad = numpy.gradient(image_array)
        #in places where gradient is 0, just make it really small
        self._vgrad[self._vgrad==0] = self.min_gradient
        self._hgrad[self._hgrad==0] = self.min_gradient
        
        
    
    def horizontal_loss(self, pixel1, label1, pixel2, label2):
        """
        pixel1 should be immediately to the left of pixel2
        """
        if (pixel1 is None) and (label1 is None):
            return 0
        w = self._w_coeff / self._hgrad[pixel1.row, pixel1.column]
        return w * self._hloss_table[self._label_set.label_to_int( label1 ), self._label_set.label_to_int( label2) ]
    
    def vertical_loss(self, pixel1, label1, pixel2, label2):
        """
        pixel1 should be immediately above pixel2
        """
        if (pixel1 is None) and (label1 is None):
            return 0
        w = self._w_coeff / self._vgrad[pixel1.row, pixel1.column]
        out = w * int(label1 == label2)
        return out
        
DataLossFunc  = GCLDataLossFunction
SmoothnessLossFunc = GCLSmoothnessLossFunction
        