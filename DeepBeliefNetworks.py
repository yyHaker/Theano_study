# -*- coding: utf-8 -*-
"""
Deep Belief Networks.

"""
import timeit
import PIL.Image as Image
import numpy
import theano
import theano.tensor as T
import os
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import tile_raster_images
from MultilayerPerceptron import HiddenLayer
from LogisticRegression import load_data, LogisticRegression
from RestrictedBoltzmannMachines import RBM


class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each other.
    The hidden layer of the RBM at layer 'i' becomes the input of the RBM at layer 'i+1'.
    The first layer RBM gets as input the input of the network, and the hidden layer of
    the last RBM represents the output. When used for classification, the DBN is treated
    as a MLP, by adding a logistic regression layer on top.
    """
    def __init(self, numpy_rng, theano_rng=None, n_ins=784, hidden_layers_sizes=[500, 500],
               n_outs=10):
        """This class is made to a variable number of layers.

        :param numpy_rng: numpy.random.RandomState, numpy random generator used to draw
         initial weights
        :param theano_rng: theano.tensor.shared_randomstreams.RandomStreams, theano random
         generator; if None, generated based on a seed draw from 'rng'
        :param n_ins: int, dimensions of input to the DBN
        :param hidden_layers_sizes: list of ints, intermediate layer size, must contain at least one value
        :param n_outs: int, dimensions of output of the network
        :return:
        """
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variable for the data
        # the data is presented as rasterized images
        self.x = T.matrix('x')

        # the labels are presented as 1D vector of [int] labels
        self.y = T.ivector('y')

        # construct the sigmoid layer
        for i in range(self.n_layers):
            # the size of the input is either the number of hidden units of the layer below or the
            # input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer if either the activation of the hidden layer below or the input
            #  of the DBN if we are on the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng, input=layer_input, n_in=input_size,
                                        n_out=hidden_layers_sizes[i], activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # here we think of the parameters of the sigmoid layer are parameters of DBN, the visible
            # biases in the RBM are parameters of RBMs, but not of the DBN.
            self.params.extend(sigmoid_layer.params)

            # construct an RBM that shared weights with this layer
            rbm_layer = RBM(input=layer_input, n_visible=input_size, n_hidden=hidden_layers_sizes[i],
                            w=sigmoid_layer.w, hbias=sigmoid_layer.b, numpy_rng=theano_rng, theano_rng=theano_rng)
            self.rbm_layers.append(rbm_layer)

        # stack one last logistic regression
        self.logLayer = LogisticRegression(input=self.sigmoid_layers[-1].output, n_in=hidden_layers_sizes[-1],
                                           n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, define as the negative log likelihood of the logistic
        # regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute errors (made on the minibatch given by self.loglayer.errors)
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        """ generates training functions for each of the RBMs.
             Generate a list of functions, for performing on step of gradient decent
        at a given layer. The function will require as input the minibatch index, and to
        train an  RBM you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :param train_set_x:
        :param batch_size:
        :param k:
        :return:
        """
        # index to a minibatch
        index = T.lscalar('index')
        learining_rate = T.scalar('learing_rate')

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
