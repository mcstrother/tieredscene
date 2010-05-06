'''
Created on May 6, 2010

@author: mcstrother
'''
import unittest
from tieredscene.lossfunctions import DataLossFunction, SmoothnessLossFunction
from tieredscene.LabelSet import LabelSet
from tieredscene.LossTable import LossTable
from tieredscene.State import State
import numpy as np

test_label_set1 = LabelSet(['M'], 'T', 'B') 

class TestDataLossFunction1(DataLossFunction.DataLossFunction):
    _label_set = test_label_set1
    _table = {'M' : [3, 0, 1, 3],
              'T' : [0, 1, 3, 3],
              'B' : [4, 4, 3, 2] }
    
    def __call__(self, pixel, label):
        return self._table[label][pixel.value]

class TestSmoothnessLossFunction1(SmoothnessLossFunction.SmoothnessLossFunction):
    _label_set = test_label_set1
    _hor_table = np.array([[0, 3, 3],
                       [3, 0, 3],
                       [1, 3, 0]])

    def vertical_loss(self, pixel1, label1, pixel2, label2):
        return 0
        
    def horizontal_loss(self, pixel1, label1, pixel2, label2):
        if pixel1 == None and label1 == None:
            return 0
        else:
            ln1 = self._label_set.label_to_int(label1)
            ln2 = self._label_set.label_to_int(label2)
            return self._hor_table[ln1, ln2]
        
class TestLossTable(unittest.TestCase):
    
    def setUp(self):
        self.image_array1 = np.array([[1, 0, 0],
                                      [2, 2, 2],
                                      [3, 3, 2]])
        self.data_func1 = TestDataLossFunction1()
        self.smoothness_func1 = TestSmoothnessLossFunction1()

    def testWholeTable(self):
        """
        Test that the LossTable is constructed properly.
        The image_array and loss functions chosen cause the
        following situations to arise:
         - The greedy choice for the first column is MMB, but the optimal
         choice is TMB.
         - The best choice for the last column based on smoothness only
         would be TMB, but taking the data loss into account it is TMM
        """
        loss_table = LossTable(self.image_array1, self.data_func1, self.smoothness_func1)
        num_states = State.count_states(self.image_array1, test_label_set1)
        print "Number of states " + str(num_states)
        print "States in order are:"
        for sn in xrange(num_states):
            print str(State.from_int(sn, test_label_set1, self.image_array1))


suite = unittest.TestLoader().loadTestsFromTestCase(TestLossTable)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()