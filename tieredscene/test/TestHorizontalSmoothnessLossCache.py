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


class TestHorizontalSmoothnessLossCache(unittest.TestCase):
    
    def setUp(self):
        self.image_array = np.array(Image.open('testimage_tiny.png').convert('L'))
        self.function = GCLSmoothnessLossFunction(self.image_array)
        self.label_set = self.function.label_set 

    def testFirstColumnLoss(self):
        """Test first column of HorizontalSmoothnessLossCache without get_loss
        """
        hor_cache = HorizontalSmoothnessLossCache(self.image_array, self.function)
        
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
                cache_loss += hor_cache.table[state.i-1, 0, 0, state.tee]
            if state.j>state.i:
                cache_loss += hor_cache.table[state.j-1, 0,0, state.el] - hor_cache.table[state.i-1, 0, 0, state.tee]
            if self.image_array.shape[0]>state.j:
                cache_loss += hor_cache.table[self.image_array.shape[0]-1, 0,0, state.tee] - hor_cache.table[state.j-1, 0,0, state.el]
            self.assertAlmostEqual(cache_loss, brute_loss)
    
    def testGetLoss(self):
        """
        Spot check get_loss
        """
        s1 = State(0,0,self.label_set.middle[0], self.label_set, self.image_array )
        hor_cache = HorizontalSmoothnessLossCache(self.image_array, self.function)
        brute_loss = 0
        for row in xrange(self.image_array.shape[0]):
            p1 = Pixel(self.image_array, 10, row)
            p2 = Pixel(self.image_array, 9, row)
            brute_loss += self.function.horizontal_loss(p1,self.label_set.bottom, p2, self.label_set.bottom)
        cache_loss = hor_cache.get_loss(s1, s1, 10)
        self.assertAlmostEqual(brute_loss, cache_loss)
        

suite = unittest.TestLoader().loadTestsFromTestCase(TestHorizontalSmoothnessLossCache)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()