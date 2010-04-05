'''
Created on Apr 4, 2010

@author: mcstrother
'''


class State(object):
    
    def __init__(self, label_set, i, j, mlabel):
        self._label_set = label_set
        self._i = i # T-M boundary
        self._j = j# M-B boundary
        self._mlabel = mlabel # the label of the middle region
        
        if not (mlabel in self._label_set.middle):
            raise ValueError("mlabel " + str(mlabel) + "not found in label_set")