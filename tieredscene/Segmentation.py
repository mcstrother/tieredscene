'''
Created on Apr 28, 2010

@author: mcstrother
'''

from tieredscene.LossTable import LossTable
import numpy

class Segmentation(object):
    
    def __init__(self, image, data_loss_func, smoothness_loss_func):
        im = image.convert('L') 
        loss_table = LossTable(numpy.array(im), data_loss_func, smoothness_loss_func )
        self._state_list = loss_table.get_optimal_state_list()
        
    def 