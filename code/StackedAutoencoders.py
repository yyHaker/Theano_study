# -*- coding: utf-8 -*-
"""
This tutorial introduces staked denoising auto-encoders using Theano.


References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
"""

import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from LogisticRegression import LogisticRegression, load_data
from MultilayerPerceptron import HiddenLayer
from Autoencoders import AutoEncoder


class StackedAutoencoder(object):
    """ Stacked denoising auto-encoder class.

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """
    def __init__(self, numpy_rng, theano_rng=None, n_ins=784, hidden_layers_sizes=[500, 500],
                 n_outs=10, corruption_levels=[0.1, 0.1]):
        """

        :param numpy_rng: numpy.random.RandomState, numpy random generator used to draw initial weights
        :param theano_rng: theano.tensor.shared_randomstreams.RandomStreams, theano random generator
        :param n_ins: int, dimension of input to the sda
        :param hidden_layers_sizes: list of ints, intermediate layer size, must contain at least one variable
        :param n_outs: int, dimension of output to the sda
        :param corruption_levels: list of float, amount of corruption to use for each layer
        """
        self.sigmoid_layers = []
        self.da_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the lables are presented as 1D vector of [int] labels

        # construct the sigmoid layer
        for i in range(self.n_layers):
            # the size of the input is either the number of hidden units of the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden layer below or the input of the sda if we are on the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input, n_in=input_size, n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # construct a denoising autoencoder that shared weights with this layer, (w, bhid)
            da_layer = AutoEncoder(numpy_rng=numpy_rng, theano_rng=theano_rng, input=layer_input, n_visible=input_size,
                                   n_hidden=hidden_layers_sizes[i], w=sigmoid_layer.w, bhid=sigmoid_layer.b)
            self.da_layers.append(da_layer)

        # we now need to add a logistic layer on top of the MLP
        self.loglayer = LogisticRegression(input=self.sigmoid_layers[-1].output, n_in=hidden_layers_sizes[-1],
                                           n_out=n_outs)
        self.params.extend(self.loglayer.params)

        self.finetune_cost = self.loglayer.negative_log_likelihood(self.y)
        self.errors = self.loglayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, learning_rate):
        """ Generate a list of functions, each of them implementing one step in training the da corresponding to
        the layer with same index. The fucntion will require as input the minibatch index, and to train a da you
        just need to iterate, calling the corresponding function on all minibatch indexes.

        :param train_set_x:
        :param batch_size:
        :param learning_rate:
        :return:
        """
        # index to a minibatch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for da in self.da_layers:
            # get the cost and update list
            cost, updates = da.get_cost_updates(corruption_level, learning_rate)

            # compile the theano function
            fn = theano.function(inputs=[index, theano.In(corruption_level, value=0.2),
                                         theano.In(learning_rate, value=0.1)], outputs=cost, updates=updates,
                                 givens={
                                     self.x: train_set_x[batch_begin: batch_end]
                                 })
            # append 'fn' to the list of functions
            pretrain_fns.append(fn)
        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """ Generate a function 'train' that implements one step of finetuning, a function 'validate' that computes
        the error on a batch from the validation set, and a function 'test' that computes the error on a batch
        from the testing set.

        :param datasets:
        :param batch_size:
        :param learning_rate:
        :return:
        """
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches fro training, validation, testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        index = T.lscalar('index')  # index to a minibatch

        # compute the gradients w.r.t the model parameters
        gparams = T.grad(self.finetune_cost, self.params)
        # compute list of fine-tuning updates
        updates = [(param, param - learning_rate * gparam)for param, gparam in zip(self.params, gparams)]

        train_fn = theano.function(inputs=[index], outputs=self.finetune_cost, updates=updates,
                                   givens={
                                       self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                       self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
                                   }, name='train')
        test_score_i = theano.function(inputs=[index], outputs=self.errors,
                                       givens={
                                           self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                           self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
                                       }, name='test')
        valid_score_i = theano.function(inputs=[index], outputs=self.errors,
                                        givens={
                                            self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                            self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                                        }, name='valid')

        # create function that scans the entire validation and test set
        def valid_socre():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_socre, test_score


def test_sda(finetune_lr=0.1, pretraining_epoches=15, pretrain_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=10):
    """ Demonstrate how to train and test a stochastic denoising autoencoder.

    :param finetune_lr:
    :param pretraining_epoches:
    :param pretrain_lr:
    :param training_epochs:
    :param dataset:
    :param batch_size:
    :return:
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print "......building the model"

    # construct the stacked denoising autoencoder class
    sda = StackedAutoencoder(numpy_rng=numpy_rng, n_ins=28 * 28, hidden_layers_sizes=[1000, 1000, 1000],
                             n_outs=10)

    print ".....pretraining the model"
    # getting the pretraining functions
    pretraining_fns = sda.pretraining_functions(train_set_x, batch_size, learning_rate=pretrain_lr)

    start_time = timeit.default_timer()
    # pretrain layer-wise
    corruption_levels = [0.1, 0.2, 0.3]
    for i in range(sda.n_layers):
        # go through pretraining_epochs
        for epoch in range(pretraining_epoches):
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index, corruption=corruption_levels[i], lr=pretrain_lr))
            print "Pre-training layer %i, epoch %d, cost %f" % (i, epoch, numpy.mean(c, dtype='float64'))
    end_time =timeit.default_timer()

    print "the pretraining code for file " + os.path.split(__file__)[1] + \
          " ran for %.2fm" % ((end_time - start_time)/60.)

    print ".....finetunning the model"
    # get the training, validation and testing function for the model
    train_fn, validate_model, test_model = sda.build_finetune_functions(datasets=datasets,
                                                                        batch_size=batch_size, learning_rate=finetune_lr)
    # early-stopping parameters
    patience = 10 * n_train_batches
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0

    while(epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                print "epoch %i, minibatch %i / %i, validation error %f %%" % \
                      (epoch, minibatch_index+1, n_train_batches, this_validation_loss*100)
                # if we got the best validation score now
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                    # test on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses, dtype='float64')
                    print "epoch %i, minibatch %i / %i, test error of best model %f %%" %\
                          (epoch, minibatch_index+1, n_train_batches, test_score * 100)
            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print "Optimization complete with best validation score of %f %%, on iteration %i, with test " \
          "performance %f %%" % (best_validation_loss * 100, best_iter + 1, test_score * 100.)
    print "the training code for file " + os.path.split(__file__)[1] + "ran for %.2fm" % ((end_time - start_time) / 60.)

if __name__ == '__main__':
    test_sda()






