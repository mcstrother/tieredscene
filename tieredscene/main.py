'''
Created on Apr 4, 2010

@author: mcstrother
'''
import sys
import optparse


def main(argv):
    usage = "%prog [options] FILE"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-m', '--middle-regions', action='store', type='int', dest='middle_regions',
                      help='the number of regions the middle section of the image should be partitioned into. default is 3', default=3 )
    (options, args) = parser.parse_args(argv)
    


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))