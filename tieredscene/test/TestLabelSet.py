'''
Created on Apr 17, 2010

@author: mcstrother
'''
import unittest
from tieredscene.LabelSet import LabelSet

class Test(unittest.TestCase):


    def setUp(self):
        self.ls1 = self.label_set = LabelSet(['1','2','3'])

    def tearDown(self):
        pass


    def test_all_labels(self):
        self.assertTrue([(num, label) for num,label in enumerate(self.ls1.all_labels)] == 
                        [(self.ls1.label_to_int(label),label) for label in self.ls1.all_labels]   )


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()