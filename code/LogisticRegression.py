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
        if y.dtype.startswith('int'):
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

    # sample data
    # print "example test label data for the first 10 example:"
    # print test_set[1][:10]

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        :param data_xy:
        :param borrow:
        :return:
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.15, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600):
    """
    Demonstrate stochastic gradient decent optimization of a log-linear model
    :param learning_rate:
    :param n_epochs:
    :param dataset:
    :param batch_size:
    :return:
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation, testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print "Build the model"

    # allocate symbolic variable for the data
    index = T.lscalar()  # index to a minibatch

    # generate symbolic variables for input (x and y represent a minibatch)
    x = T.matrix('x')
    y = T.ivector('y')

    # construct the logistic regression class
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index], outputs=classifier.errors(y),
                                 givens={
                                     x: test_set_x[index*batch_size: (index+1)*batch_size],
                                     y: test_set_y[index*batch_size: (index+1)*batch_size]
                                 })
    validate_model = theano.function(inputs=[index], outputs=classifier.errors(y),
                                     givens={
                                         x: valid_set_x[index*batch_size: (index+1)*batch_size],
                                         y: valid_set_y[index*batch_size: (index+1)*batch_size]
                                     })
    # compute the gradient of cost with respect to theta = (w, b)
    g_w = T.grad(cost=cost, wrt=classifier.w)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.w, classifier.w - learning_rate*g_w), (classifier.b, classifier.b - learning_rate*g_b)]

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={
                                      x: train_set_x[index*batch_size: (index+1)*batch_size],
                                      y: train_set_y[index*batch_size: (index+1)*batch_size]
                                  })
    print "...train the model"
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best if found
    improvement_threshold = 0.995  # a relative improvement of this is considered significant
    # go through this many minibatch before checking the network on the validation set
    validation_frequency = min(n_train_batches, patience//2)

    best_validation_loss = numpy.inf
    test_score = 0.

    start_time = timeit.default_timer()

    done_looping = False  # is stop loop
    epoch = 0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print "epoch %i , minibatch  %i / %i, validation error %f %%" % \
                      (epoch, minibatch_index+1, n_train_batches, this_validation_loss * 100.)

                # if we get the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print "epoch %i, minibatch %i / %i , test error of best model %f %%" % \
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.)

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print "optimization complete with best validation score of %f %%  with test performance %f %%" % \
          (best_validation_loss * 100, test_score * 100)
    print "The code run for %d epochs, with %f epochs/sec" % (epoch, 1. * epoch / (end_time - start_time))
    print 'The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)


def predict():
    """
     load a trained model and use it to predict labels.
    :return:
    """
    # load the save model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(inputs=[classifier.input], outputs=classifier.y_pred)

    # we can test it on some examples from test test
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predict_values = predict_model(test_set_x[:10])
    print "predict values for the first example in test set:"
    print predict_values


if __name__ == "__main__":
    sgd_optimization_mnist()
    # predict()






















