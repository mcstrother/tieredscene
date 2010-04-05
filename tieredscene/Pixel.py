'''
Created on Apr 4, 2010

@author: mcstrother
'''

class Pixel(object):
    """
    Object representing a pixel in an image
    """
    
    def __init__(self, image, column, row):
        self._image = image
        self._column = column
        self._row = row