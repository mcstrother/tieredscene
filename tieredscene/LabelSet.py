'''
Created on Apr 4, 2010

@author: mcstrother
'''
from collections import set


class LabelSet(object):
    
    def __init__(self, top_label, bottom_label, middle_labels):
        self._top = top_label
        self._bottom = bottom_label
        self._middle = set(middle_labels)
        
    @property
    def top(self):
        return self._top
    
    @property
    def bottom(self):
        return self._bottom
    
    @property
    def middle(self):
        return self._middle
    
    