'''
Created on May 1, 2010

@author: mcstrother
'''
import unittest
import Image
import numpy as np
from tieredscene.GeometricClassLabeling import GCLSmoothnessLossFunction
from tieredscene.VerticalSmoothnessLossCache import VerticalSmoothnessLossCache
from tieredscene.State import State
from tieredscene.Pixel import Pixel

class TestSmoothnessLossCaches(unittest.TestCase):



    def testVerticalGetLoss(self):
        """Test that VerticalSmoothnessLossCache returns the same loss
        for every state as we would get by calculating it by brute force
        using the DataLossFunction
        """
        self.image_array = np.array(Image.open('testimage_tiny.png').convert('L'))
        self.function = GCLSmoothnessLossFunction(self.image_array)
        self.vert_cache = VerticalSmoothnessLossCache(self.image_array, self.function)
        self.label_set = self.function.label_set 
        for column in xrange(self.image_array.shape[1]):
            for state_num in xrange(State.count_states(self.image_array, self.label_set)):
                state = State.from_int(state_num, self.label_set, self.image_array)
                brute_loss = self.function.vertical_loss(None, None, Pixel(self.image_array,column , 0), state.get_row_label(0))
                for row in xrange(1, state.i):
                    p1 = Pixel(self.image_array, column, row-1)
                    l1 = state.get_row_label(row-1)
                    p2 = Pixel(self.image_array, column, row)
                    l2 = state.get_row_label(row)
                    brute_loss += self.function(p1, l1, p2 ,l2)
                cache_loss = self.vert_cache.get_loss(state, column)
                self.assertAlmostEquals(brute_loss, cache_loss)
            
suite = unittest.TestLoader().loadTestsFromTestCase(TestSmoothnessLossCaches)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()