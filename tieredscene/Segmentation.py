'''
Created on Apr 28, 2010

@author: mcstrother
'''

from tieredscene.LossTable import LossTable
import numpy
import logging

class Segmentation(object):
    
    def __init__(self, image_array, data_loss_func, smoothness_loss_func):
        loss_table = LossTable(image_array, data_loss_func, smoothness_loss_func )
        self._state_list = loss_table.get_optimal_state_list()
        for state in self._state_list:
            logging.debug(state.__str__())
        
    def to_array(self):
        out = [state.to_array() for state in self._state_list]
        out = numpy.array(out)
        out = out.transpose()
        return out