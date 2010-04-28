'''
Created on Apr 4, 2010

@author: mcstrother
'''
import sys
import optparse
import Image
from tieredscene import GeometricClassLabeling, Segmentation
import numpy

def main(argv):
    usage = "%prog [options] FILE"
    parser = optparse.OptionParser(usage=usage)
    #parser.add_option('-m', '--middle-regions', action='store', type='int', dest='middle_regions',
    #                  help='the number of regions the middle section of the image should be partitioned into. default is 3', default=3 )
    (options, args) = parser.parse_args(argv)
    image_file_name = args[0]
    image_array = numpy.array(Image.open(image_file_name, 'L'))
    data_loss_function = GeometricClassLabeling.GCLDataLossFunction()
    smoothness_loss_function = GeometricClassLabeling.GCLSmoothnessLossFunction(image_array)
    segmentation = Segmentation.Segmentation(image_array, data_loss_function, smoothness_loss_function)
    

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))