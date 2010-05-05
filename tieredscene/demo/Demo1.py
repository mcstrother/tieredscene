'''
Created on Apr 28, 2010

@author: mcstrother
'''
from tieredscene import main

if __name__=='__main__':
    argv = '-o demo1_out.png -l Simple ../test/testimage_smallest.png'.split()
    main.main(argv)