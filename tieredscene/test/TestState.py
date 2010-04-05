'''
Created on Apr 5, 2010

@author: mcstrother
'''
import unittest
import numpy
from tieredscene.State import State
from tieredscene.LabelSet import LabelSet

class TestState(unittest.TestCase):


    def setUp(self):
        self.label_set = LabelSet(['1','2','3'])
        self.image_array = numpy.zeros((3,5))
        
        

    def test_as_int(self):
        test_list =[]
        for el in self.label_set.middle:
            for i in xrange(0, 3):
                for j in xrange(i,3):
                    state = State(i, j, el, self.label_set, self.image_array)
                    test_list.append(state.as_int())
        self.assertEqual(test_list, range(18))
    
    def test_from_int(self):
        test_list = []
        for x in range(18):
            state = State.from_int(x, self.label_set, self.image_array)
            test_list.append(state.as_int())
        self.assertEqual(test_list, range(18))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()