# -*- coding: utf-8 -*-
"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy
import theano
import theano.tensor as T


class LogisticRegression(object):
    """ Multi-class Logistic Regression Class

     The Logistic regression is fully described by a weight matrix : math:'w'
     and bias vector  :math:'b'. Classification is done by projecting data
     points onto a set of hyperplanes, the distance to which is used to determine
     a class membership probability.
    """
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :param input: type: theano.tensor.TensorType, symbolic variable that describes
          the input of the architecture (one minibatch)
        :param n_in: int, number of input units, the dimension of the space in which the datapoints lie
        :param n_out: int, number of output units, the dimension of the space in which the labels lie
        """
        # initialize with 0 the weights w as a matrix of shape (n_in, n_out)
        self.w = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='w', borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        # symbolic expression for computing the matrix of class-membership probabilities
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.w) + self.b)
        # symbolic description of how to compute prediction as class whose probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # parameters of the model
        self.params = [self.w, self.b]
        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """ Return the mean of negative of log-likelihood of the prediction of the model under a given
         target distribution.
        :param y:
        :return:
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """ Return a float representing the number of errors in the minibatch over the total number of examples
        of the minibatch.

        :param y: theano.tensor.TensorType, correspond to a vector that gives for each example the correct label
        :return:
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startwith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError


def load_data(dataset):
    """ Loads the dataset

    :param dataset: string, the path to the dataset(here MNIST)
    :return:
    """
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    # check if dataset is in the data directory
    if data_dir == "" and not os.path.exists(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Download data from %s' % origin
        urllib.request.urlretrieve(origin, dataset)
    print '... loading data'

    # load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        :param data_xy:
        :param borrow:
        :return:
        """
        data_x, data_y = data_xy
        shard_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval
























