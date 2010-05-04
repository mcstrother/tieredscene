'''
Created on May 1, 2010

@author: mcstrother
'''
import unittest
import numpy as np
from tieredscene.GeometricClassLabeling import GCLSmoothnessLossFunction
from tieredscene.HorizontalSmoothnessLossCache import HorizontalSmoothnessLossCache
from tieredscene.State import State
from tieredscene.Pixel import Pixel
import Image


def _get_brute_loss(function, image_array, col, s1, s2):
    """Calculate the horizontal smoothness loss
    by brute force.
    """
    brute_loss = 0
    for row in xrange(image_array.shape[0]):
        p1 = Pixel(image_array, 10, row)
        p2 = Pixel(image_array, 9, row)
        l1 = s1.get_row_label(row)
        l2 = s2.get_row_label(row)
        brute_loss += function.horizontal_loss(p1,l1, p2, l2)
    return brute_loss

class TestHorizontalSmoothnessLossCache(unittest.TestCase):
    
    def setUp(self):
        self.image_array = np.array(Image.open('tiny_texture.png').convert('L'))
        self.function = GCLSmoothnessLossFunction(self.image_array)
        self.label_set = self.function.label_set
        self.s1 = State(0,0,self.label_set.middle[0], self.label_set, self.image_array )
        self.s2 = State(3, 7, self.label_set.middle[1], self.label_set, self.image_array)
        self.s3 = State(2, 5, self.label_set.middle[0], self.label_set, self.image_array)
        self.s4 = State(0, 1, self.label_set.middle[0], self.label_set, self.image_array)
        self.hor_cache =  HorizontalSmoothnessLossCache(self.image_array, self.function)
        
        
    def testFirstColumnLoss(self):
        """Test first column of HorizontalSmoothnessLossCache without get_loss
        """
        
        
        #check first column
        for state_num in xrange(State.count_states(self.image_array, self.label_set)):
            state = State.from_int(state_num, self.label_set, self.image_array)
            brute_loss = 0
            for row in xrange(self.image_array.shape[0]):
                p2 = Pixel(self.image_array, 0, row)
                l2 = state.get_row_label(row)
                brute_loss += self.function.horizontal_loss(None, None, p2, l2)
            cache_loss = 0
            if state.i>0:
                cache_loss += self.hor_cache.table[state.i-1, 0, 0, state.tee]
            if state.j>state.i:
                cache_loss += self.hor_cache.table[state.j-1, 0,0, state.el] - self.hor_cache.table[state.i-1, 0, 0, state.tee]
            if self.image_array.shape[0]>state.j:
                cache_loss += self.hor_cache.table[self.image_array.shape[0]-1, 0,0, state.tee] - self.hor_cache.table[state.j-1, 0,0, state.el]
            self.assertAlmostEqual(cache_loss, brute_loss)
    
    def testGetLoss1(self):
        """Test HorizontalSmoothnessLossCache with ib = jb= i = j = 0
        """
        brute_loss = _get_brute_loss(self.function, self.image_array, 10, self.s1, self.s1)
        cache_loss = self.hor_cache.get_loss(self.s1, self.s1, 10)
        self.assertAlmostEqual(brute_loss, cache_loss)
    
    def testGetLoss2(self):
        """Test HorizontalSmoothnessLossCache with ib = jb = 0 <  i < j
        """
        brute_loss = _get_brute_loss(self.function, self.image_array, 10, self.s1, self.s2)
        cache_loss = self.hor_cache.get_loss(self.s1, self.s2, 10)
        self.assertAlmostEqual(cache_loss, brute_loss)
    
    def testGetLoss3(self):
        """Test HorizontalSmoothnessLossCache with relative_positioning 1
        """
        brute_loss = _get_brute_loss(self.function, self.image_array, 10, self.s2, self.s3)
        cache_loss = self.hor_cache.get_loss(self.s2, self.s3, 10)
        self.assertAlmostEqual(cache_loss, brute_loss)
    
    def testGetLoss4(self):
        """Test HorizontalSmoothnessLossCache with relative_positioning 4
        """
        brute_loss = _get_brute_loss(self.function, self.image_array, 10, self.s3, self.s2)
        cache_loss = self.hor_cache.get_loss(self.s3, self.s2, 10)
        self.assertAlmostEqual(cache_loss, brute_loss)
    
    def testGetLoss5(self):
        """Test HorizontalSmoothnessLossCache with relative_positioning 5
        """
        brute_loss = _get_brute_loss(self.function, self.image_array, 10, self.s4, self.s2)
        cache_loss = self.hor_cache.get_loss(self.s4, self.s2, 10)
        self.assertAlmostEqual(cache_loss, brute_loss)
    
    def testGetLoss6(self):
        """Test HorizontalSmoothnessLossCache with relative_positioning 0
        """
        brute_loss = _get_brute_loss(self.function, self.image_array, 10, self.s2, self.s4)
        cache_loss = self.hor_cache.get_loss(self.s2, self.s4, 10)
        self.assertAlmostEqual(cache_loss, brute_loss)
    

suite = unittest.TestLoader().loadTestsFromTestCase(TestHorizontalSmoothnessLossCache)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()