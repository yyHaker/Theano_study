# -*- coding: utf-8 -*-
"""
Here we implement a LeNet model in Theano.
-------------------------------------------

This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.

This implementation simplifies the model in the following ways:
 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer


References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
"""
import os
import sys
import timeit

import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

from LogisticRegression import LogisticRegression, load_data
from MultilayerPerceptron import HiddenLayer


class LeNetConvPoolLayer(object):
    """ Pool layer of a convolution network, which contains a {convolution + max-pooling} layer.

    """
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """

        :param rng: numpy.random.RandomState, a random number generator used to initialize weights
        :param input: theano.tensor.dtensor4, symbolic image tensor, of shape image_shape
        :param filter_shape: tuple or list of length 4, (number of filters, num input feature maps,
                filter height, filter width)
        :param image_shape: tuple or list of length 4, (batch size, num input feature maps,
                image height, image width)
        :param poolsize: tuple or list of length 2, (the downsampling (pooling) factor) (#rows, #cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:])) // numpy.prod(poolsize)

        # initialize weights with random weights
        w_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.w = theano.shared(value=numpy.asarray(
            rng.uniform(low=-w_bound, high=w_bound, size=filter_shape), dtype=theano.config.floatX
        ), borrow=True)

        # the bias is 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0], ), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # convolve input feature  maps with filters
        conv_out = conv2d(input=input, filters=self.w, input_shape=image_shape, filter_shape=filter_shape)

        # pooling each map individually, using maxpooling
        pooled_out = pool.pool_2d(input=conv_out, ws=poolsize, ignore_border=True)

        # dimshuffle the b to (1, n_filters, 1, 1)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # parameters
        self.params = [self.w, self.b]

        # keep track of model input
        self.input = input


def evaluate_LeNet5(learing_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[20, 50], batch_size=500):
    """ Demonstrate LeNet5 on MNIST dataset.

    :param learing_rate: float
    :param n_epoches: int, maximal number of epoches to run the optimizer
    :param dataset: string, path to the dataset used for training/testing
    :param nkerns: list of ints, number of kernels on each layer
    :param batch_size:
    :return:
    """
    rng = numpy.random.RandomState(23445)
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')
    y = T.ivector('y')

    print "...............building the model"

    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1, 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 1, 28, 28),
                                filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus  of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size, nkerns[0], 12, 12),
                                filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

    # The HiddenLayer being fully-connected, it operates on 2D matrices of shape (batch_size, num_pixels)
    layer2_input = layer1.output.flatten(2)  # (500, 50 *4 *4)

    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1]*4*4, n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoid layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the  cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that  are made by the model
    test_model = theano.function(inputs=[index], outputs=layer3.errors(y),
                                 givens={
                                     x: test_set_x[index * batch_size: (index+1) * batch_size],
                                     y: test_set_y[index * batch_size: (index+1) * batch_size]
                                 })
    validate_model = theano.function(inputs=[index], outputs=layer3.errors(y),
                                     givens={
                                         x: valid_set_x[index * batch_size: (index +1) * batch_size],
                                         y: valid_set_y[index * batch_size: (index +1) * batch_size]
                                     })

    # create a list of gradients for all model parameters to be fit by gradient decent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [(param, param - learing_rate * gparam)for param, gparam in zip(params, grads)]

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={
                                      x: train_set_x[index*batch_size: (index+1)*batch_size],
                                      y: train_set_y[index*batch_size: (index+1)*batch_size]
                                  })

    # Train model
    print ".....training"
    # early stop parameters
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))


if __name__ == '__main__':
    evaluate_LeNet5(learing_rate=0.3, batch_size=45)

    #  learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=45 : validation 0.900901%, test 0.760761%, time 27.65m
    #  learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=45 : validation 0.930931%, test 0.810811%, time 13.87m
    #  learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=45 : validation 0.900901%, test 0.780781%, time 27.62m
    #  learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=45 : validation 1.001001%, test 0.790791%, time 3.92m
    # learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=45 : validation 0.940941%, test 0.770771%, time 15.35m
    # learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=45 : validation 0.960961%, test 0.790791%, time 7.02m
    #  learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=50 : validation 0.91%, test 0.81%, time 6.55m
    #  learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=50 : validation 0.91%, test 0.80%, time 7.91m
    #  learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=100 : validation 0.89%, test 0.82%, time 18.51m
    #  learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=500 : validation 0.94%, test 0.83%, time 19.84m
    #  learning rate=0.3 n_epoches=200, nkerns=[20, 50], batch_size=500 : validation 0.94%, test 0.85%, time 19.73m
    # learning rate=0.1 n_epoches=200, nkerns=[20, 50], batch_size=500 : validation 0.98%, test 0.9%, time 19.75m
    # learning rate=0.01 n_epoches=200, nkerns=[20, 50], batch_size=500 : validation 1.37%, test 1.19%, time 20.30m
