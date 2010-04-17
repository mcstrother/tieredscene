'''
Created on Apr 5, 2010

@author: mcstrother
'''


from tieredscene import DataLossFunction
from tieredscene import SmoothnessLossFunction
from tieredscene import LabelSet
import numpy

gcl_label_set = LabelSet.LabelSet(['L','C','R'], 'T', 'B')


class GCLDataLossFunction(DataLossFunction.DataLossFunction):
    _label_set = gcl_label_set
    
    def __call__(self, pixel, label):
        raise NotImplementedError("TODO: fill this in using the per pixel class confidences from reference 9 from the paper")
    
    

class GCLSmoothnessLossFunction(SmoothnessLossFunction.SmoothnessLossFunction):
    _label_set = gcl_label_set
    
    #TODO: calculate the w term
    def __init__(self, image_array):
        SmoothnessLossFunction.SmoothnessLossFunction.__init__(self)
        self._hloss_table = numpy.array([[0, 1, 4, 1, 3 ],
                                         [1, 0, 1, 1, 1 ],
                                         [4, 1, 0, 1, 1 ],
                                         [1, 1, 1, 0, 2 ],
                                         [1, 1, 3, 2, 0 ]
                                         ])
        self._vgrad, self._hgrad = numpy.gradient(image_array)
        self._w_coeff = .1 #TODO: may need to be tweaked
        
    
    def horizontal_loss(self, pixel1, label1, pixel2, label2):
        """
        pixel1 should be immediately to the left of pixel2
        """
        if (pixel1 is None) and (label1 is None):
            return 0
        w = self._w_coeff / self._hgrad[pixel1.row, pixel1.column]
        return w * self._hloss_table[self._label_set.to_int( label1 ), self._label_set.to_int( label2) ]
    
    def vertical_loss(self, pixel1, label1, pixel2, label2):
        """
        pixel1 should be immediately above pixel2
        """
        if (pixel1 is None) and (label1 is None):
            return 0
        w = self._w_coeff / self._vgrad[pixel1.row, pixel1.column]
        return w * int(label1 == label2)
        
        
        