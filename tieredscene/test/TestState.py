'''
Created on Apr 5, 2010

@author: mcstrother
'''
import unittest
import numpy
from tieredscene.State import State
from tieredscene.LabelSet import LabelSet

class Test(unittest.TestCase):


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
        
        #Test State.count_states
        state_count = State.count_states(self.image_array, self.label_set)
        expected_state_count = max(test_list)+1
        self.assertEqual(expected_state_count, state_count,
                         msg = "State.count_states returns an incorrect number of states.  Got " + str(state_count) + 
                         ", expected " + str(expected_state_count) + '.' )
    
    def test_from_int(self):
        test_list = []
        for x in range(18):
            state = State.from_int(x, self.label_set, self.image_array)
            test_list.append(state.as_int())
        self.assertEqual(test_list, range(18))
        
        
    def test_get_row_label(self):
        #Test 1
        s1 = State(0, 0, self.label_set.middle[0], self.label_set, self.image_array)
        label_list = [self.label_set.bottom for i in xrange(self.image_array.shape[0])]
        for row in xrange(len(label_list)):
            self.assertEqual(s1.get_row_label(row), label_list[row])
        #Test 2
        s2 = State(0,1, self.label_set.middle[0], self.label_set, self.image_array )
        label_list = [self.label_set.middle[0]] + [self.label_set.bottom for i in xrange(self.image_array.shape[0]-1)]
        for row in xrange(len(label_list)):
            self.assertEqual(s2.get_row_label(row), label_list[row])
        #Test 3
        s3 =  State(1,1, self.label_set.middle[0], self.label_set, self.image_array )
        label_list = [self.label_set.top] + [self.label_set.bottom for i in xrange(self.image_array.shape[0]-1)]
        for row in xrange(len(label_list)):
            self.assertEqual(s3.get_row_label(row), label_list[row])
        #Test 4
        s4 = State(1,3, self.label_set.middle[1], self.label_set, self.image_array )
        label_list = [self.label_set.top, self.label_set.middle[1], self.label_set.middle[1]] + [self.label_set.bottom for i in xrange(self.image_array.shape[0]-4)]
        for row in xrange(len(label_list)):
            self.assertEqual(s4.get_row_label(row), label_list[row])
        
suite = unittest.TestLoader().loadTestsFromTestCase(Test)
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()