'''
Created on May 4, 2010

@author: mcstrother
'''
import unittest
from tieredscene.lossfunctions.Simple import DataLossFunc
import Image
import numpy as np
from tieredscene.DataLossCache import DataLossCache
from tieredscene.State import State
from tieredscene.Pixel import Pixel

class TestDataLossCache(unittest.TestCase):


    def setUp(self):
        self.image_array = np.array(Image.open('tiny_texture.png').convert('L'))
        self.function = DataLossFunc()
        self.label_set = self.function.label_set 


    def testGetLoss(self):
        column = 4
        data_cache = DataLossCache(self.image_array, self.function)
        for state_num in xrange(State.count_states(self.image_array, self.label_set)):
            state = State.from_int(state_num, self.label_set, self.image_array)
            brute_loss = 0
            for row in xrange(self.image_array.shape[0]):
                label = state.get_row_label(row)
                pixel = Pixel(self.image_array, column, row)
                brute_loss += self.function(pixel, label)
            cache_loss = data_cache.get_loss(state, column)
            self.assertAlmostEqual(brute_loss, cache_loss)
                
            
suite = unittest.TestLoader().loadTestsFromTestCase(TestDataLossCache)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()