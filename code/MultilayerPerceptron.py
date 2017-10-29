# -*- coding: utf-8 -*-
"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5
"""
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T

from LogisticRegression import LogisticRegression, load_data


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, w=None, b=None, activation=T.tanh):
        """ Typical hidden layer of a MLP: units are fully-connected and have sigmoid activation of function.
        Weight matrix w is of shape(n_in, n_out) and the bias vector b is of shape (n_out, ).

        :param rng: numpy.random.RandomState, a random number generator used to initialize weights
        :param input: theano.tensor.dmatrix, a symbolic tensor of shape (n_examples, n_in)
        :param n_in: int, dimensionality of input
        :param n_out: int, number of hidden units
        :param w:
        :param b:
        :param activation: theano.op or function, Non-Linearity to be applied in the hidden layer
        """
        self.input = input

        if w is None:
            w_values = numpy.asarray(
                rng.uniform(size=(n_in, n_out), low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out))),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                w_values *= 4
            w = theano.shared(value=w_values, name='w', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out, ), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.w = w
        self.b = b

        line_output = T.dot(input, self.w) + self.b
        self.output = line_output if activation is None else activation(line_output)

        # parameters of the model
        self.params = [self.w, self.b]


class MLP(object):
    """ Multi-layer Perceptron Class

        A multilayer perceptron is a feedforward artificial neural network model that has one layer or
    more of hidden units and nonlinear activations. Intermediate layers usually have as activation function
    tanh or the sigmoid function.
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """ Initialize the parameters for the multilayer perceptron.

        :param rng: numpy.random.RandomState, a random number to initialize weights
        :param input: theano.tensor.TensorType, symbolic variable that describe the input of the architecture (one minibatch)
        :param n_in: int, number of input units, the dimension of space in which the datapoints lie
        :param n_hidden: int, number of hidden units
        :param n_out: int, number of output units, the dimension of the space in which the labels lie
        """
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)

        # The logistic regression layer gets as input the hidden units of the hidden layer
        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)

        # L1 norm
        self.L1 = abs(self.hiddenLayer.w).sum() + abs(self.logRegressionLayer.w).sum()
        # L2 norm
        self.L2_sqr = (self.hiddenLayer.w**2).sum() + (self.logRegressionLayer.w**2).sum()

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input


def sgd_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=18, n_hidden=1000 ):
    """ Demonstrate stochastic gradient decent optimization for a multilayer perceptron.

    :param learning_rate: float
    :param L1_reg: float, L1-norm's weight when added to the cost
    :param L2_reg: float, l2-norm's weight when added to the cost
    :param n_epochs: int, maximal number of epoches to run the optimizer
    :param dataset: string, the path of the MNIST dataset file
    :param batch_size:
    :param n_hidden:
    :return:
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print ".....Building the model"

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a minibatch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of int labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=n_hidden, n_out=10)
    # the cost we minimize during training is the negative log likelihood of the model plus
    # the regularization terms (L1 and L2);
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made by the model on a minibatch
    test_model = theano.function(inputs=[index], outputs=classifier.errors(y),
                                 givens={
                                     x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: test_set_y[index * batch_size: (index + 1) * batch_size]
                                 })
    validate_model = theano.function(inputs=[index], outputs=classifier.errors(y),
                                     givens={
                                         x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                                     })

    # compute the gradient of cost with respect to theta (stored in params)
    gparams = [T.grad(cost, param) for param in classifier.params]

    # update the params
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]

    # compiling a Theano function 'train_model' that returns the cost, but in the same time updates the
    # parameter of the model based on the rules defined in 'updates'
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={
                                      x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                      y: train_set_y[index * batch_size: (index + 1) * batch_size]
                                  })
    print "...training"

    # early stop parameters
    patience = 20000
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995
    validation_frequence = min(n_train_batches, patience // 2)


    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequence == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print "epoch %i, minibatch %i / %i, validation error %f %%" \
                      % (epoch, minibatch_index+1, n_train_batches, this_validation_loss * 100)
                # if we got the best validation score now
                if this_validation_loss < best_validation_loss:
                    # improvement patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print "epoch %i, minibatch %i / %i, test error of best model %f %%" % \
                          (epoch, minibatch_index+1, n_train_batches, test_score * 100)

            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print "Optimization complete. Best validation score of %f %% obtained at iteration %i, with test " \
          "performance %f %%" % (best_validation_loss * 100, best_iter + 1, test_score * 100)
    print "the code for file " + os.path.split(__file__)[1] + " ran for %.2fm" % ((end_time - start_time)/60.)


if __name__ == '__main__':
    print "begin....."
    sgd_mlp()











