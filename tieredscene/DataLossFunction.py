'''
Created on Apr 4, 2010

@author: mcstrother
'''


class DataLossFunction(object):
    
    def __call__(self, pixel, label):
        """
        """
        raise NotImplementedError("Subclasses of DataLossFunction must override __call__")


