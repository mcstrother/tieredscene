'''
Created on Apr 4, 2010

@author: mcstrother
'''
import sys
import optparse
import Image
import numpy
import logging
import matplotlib.pyplot as plt
import Segmentation

def init_logging():
    numpy.set_printoptions(threshold=numpy.nan)
    logging.basicConfig(stream = sys.stdout, level = logging.DEBUG)

def parse_args(argv):
    usage = "%prog [options] FILE"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-o', action='store', dest='output_name', default = 'segmentation.png' ,
                      help='the name of the file the segmented image should be written to')
    parser.add_option('-l', action = 'store', dest ='loss_function', default = 'GeometricClassLabeling',
                      help='name of a .py file in tieredscene.lossfunctions package with loss function definitions')
    #parser.add_option('-m', '--middle-regions', action='store', type='int', dest='middle_regions',
    #                  help='the number of regions the middle section of the image should be partitioned into. default is 3', default=3 )
    (options, args) = parser.parse_args(argv)
    return (options, args)

def get_image_array(args):
    image_file_name = args[0]
    logging.debug('Opening image...')
    im = Image.open(image_file_name)
    im = im.convert('L')
    return numpy.array(im)

def main(argv):
    init_logging()
    (options, args) = parse_args(argv)
    function_module = __import__('tieredscene.lossfunctions.' + options.loss_function, fromlist = ['DataLossFunc','SmoothnessLossFunction'])
    print function_module.__name__
    image_array = get_image_array(args)
    data_loss_function = function_module.DataLossFunc()
    smoothness_loss_function = function_module.SmoothnessLossFunc(image_array)
    segmentation = Segmentation.Segmentation(image_array, data_loss_function, smoothness_loss_function)
    plt.imsave(options.output_name, segmentation.to_array())
    #out = Image.fromarray(segmentation.to_array())
    #out.save(options.output_name)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))