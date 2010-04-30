'''
Created on Apr 4, 2010

@author: mcstrother
'''
import sys
import optparse
import Image
from tieredscene import GeometricClassLabeling, Segmentation
import numpy
import logging

def init_logging():
    logging.basicConfig(stream = sys.stdout, level = logging.DEBUG)

def main(argv):
    init_logging()
    usage = "%prog [options] FILE"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-o', action='store', dest='output_name', default = 'segmentation.png' ,
                      help='the name of the file the segmented image should be written to')
    #parser.add_option('-m', '--middle-regions', action='store', type='int', dest='middle_regions',
    #                  help='the number of regions the middle section of the image should be partitioned into. default is 3', default=3 )
    (options, args) = parser.parse_args(argv)
    image_file_name = args[0]
    logging.debug('Opening image...')
    im = Image.open(image_file_name)
    im = im.convert('L')
    image_array = numpy.array(im)
    logging.debug('Image opened and converted to array.')
    data_loss_function = GeometricClassLabeling.GCLDataLossFunction()
    smoothness_loss_function = GeometricClassLabeling.GCLSmoothnessLossFunction(image_array)
    logging.debug('Creating segmentation object')
    segmentation = Segmentation.Segmentation(image_array, data_loss_function, smoothness_loss_function)
    logging.debug('Segmentation object created.')
    out = Image.fromarray(segmentation.to_array())
    out.save(options.output_name)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))