# -*- coding: utf-8 -*-
"""
ConvOp is the main workhorse for implementing a convolutional layer in Theano. ConvOp is
 used by theano.tensor.signal.conv2d, which takes two symbolic inputs:
   - a 4D tensor corresponding to a mini-batch of input images. The shape of the tensor is
   as follows: [mini-batch size, number of input feature maps, image height, image width].
   -a 4D tensor corresponding to the weight matrix W. The shape of the tensor
    is: [number of feature maps at layer m, number of feature maps at layer m-1,
    filter height, filter width]

Below is the Theano code for implementing a convolutional layer.  The input consists of 3
 features maps (an RGB color image) of size 120x160. We use two convolutional filters
 with 9x9 receptive fields.
"""
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy

import pylab
from PIL import Image

rng = numpy.random.RandomState(23455)

# initial 4D tensor for input
input = T.tensor4(name='input')

# initialize shared variable for weights
w_shp = (2, 3, 9, 9)
w_bound = numpy.sqrt(3 * 9 * 9)
w = theano.shared(numpy.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=w_shp),
                                dtype=input.dtype), name='w')
# initialize shared variable for bisas (1D tensor) with random values
b_shp = (2, )
b = theano.shared(numpy.asarray(rng.uniform(low=-.5, high=.5, size=b_shp), dtype=input.dtype), name='b')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv2d(input, w)

# build symbolic expression to add bias and apply activation action
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# create theano function to compute filtered images
f = theano.function([input], output)

# open random image of dimensions 516x639
img = Image.open('../doc/images/3wolfmoon.jpg')
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float64') / 256.
# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_img = f(img_)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img); pylab.gray()

pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()
