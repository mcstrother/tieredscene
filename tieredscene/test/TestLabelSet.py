'''
Created on Apr 17, 2010

@author: mcstrother
'''
import unittest
from tieredscene.LabelSet import LabelSet

class Test(unittest.TestCase):


    def setUp(self):
        self.ls1 = self.label_set = LabelSet(['a','b','c'])

    def tearDown(self):
        pass


    def test_all_labels(self):
        #ensure the correct number of labels are returned
        label_list = [label for label in self.ls1.all_labels]
        self.assertEqual(len(label_list), 5)
        #ensure labels are returns in the correct order
        self.assertTrue(['a','b','c','T','B'] == label_list)
    
    def test_label_to_int(self):
        #ensure label_to_int returns the correct int for all labels
        self.assertTrue([(num, label) for num,label in enumerate(self.ls1.all_labels)] == 
                        [(self.ls1.label_to_int(label),label) for label in self.ls1.all_labels] )


suite = unittest.TestLoader().loadTestsFromTestCase(Test)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()