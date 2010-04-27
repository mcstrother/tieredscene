'''
Created on Apr 17, 2010

@author: mcstrother
'''
import unittest
from tieredscene.test import TestLabelSet, TestState

if __name__ == '__main__':
    all_tests = unittest.TestSuite([TestLabelSet.suite, TestState.suite])
    unittest.TextTestRunner(verbosity=2).run(all_tests)