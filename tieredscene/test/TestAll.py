'''
Created on Apr 17, 2010

@author: mcstrother
'''
import unittest
from tieredscene.test import TestLabelSet, TestState, TestVerticalSmoothnessLossCache, TestHorizontalSmoothnessLossCache, TestDataLossCache
import numpy

if __name__ == '__main__':
    numpy.set_printoptions(threshold=numpy.nan)
    all_tests = unittest.TestSuite([TestLabelSet.suite, 
                                    TestState.suite, 
                                    TestVerticalSmoothnessLossCache.suite,
                                    TestHorizontalSmoothnessLossCache.suite,
                                    TestDataLossCache.suite
                                    ])
    unittest.TextTestRunner(verbosity=2).run(all_tests)