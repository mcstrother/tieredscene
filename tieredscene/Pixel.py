'''
Created on Apr 4, 2010

@author: mcstrother
'''

class Pixel(object):
    """
    Object representing a pixel in an image
    """
    
    def __init__(self, image_array, column, row):
        self._image_array = image_array
        self._column = column
        self._row = row
        
    @property
    def column(self):
        return self._column
    
    @property
    def row(self):
        return self._row