# -*- coding: utf-8 -*-
"""
Implement the Recurrent Neural Network using BPTT algorithm.



refering from: http://www.wildml.com/2015/09/recurrent-neural-networks
-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
"""
import numpy as np
import theano as theano
import theano.tensor as T
import operator

import csv
import itertools
import nltk
import sys
import os
import time
from datetime import datetime


class RNN(object):
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        """

        :param word_dim: the size of our vocabulary
        :param hidden_dim: the size of our hidden layer
        :param bptt_truncate:
        """
        # assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # randomly initialize the network parameters
        U = np.random.uniform(low=-np.sqrt(1./word_dim), high=np.sqrt(1./word_dim), size=(hidden_dim, word_dim))
        V = np.random.uniform(low=-np.sqrt(1./hidden_dim), high=np.sqrt(1./hidden_dim), size=(word_dim, hidden_dim))
        W = np.random.uniform(low=-np.sqrt(1./hidden_dim), high=np.sqrt(1./hidden_dim), size=(hidden_dim, hidden_dim))

        # create shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))

        # store the theano graph
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.ivector('x')
        y = T.ivector('y')

        # each time propagation step
        def forward_prop_step(x_t, s_t_pre, U, V, W):
            # Note that we are indexing U by x_t. This is the same as multiplying U with a one-hot vector.
            s_t = T.tanh(U[:, x_t] + T.dot(W, s_t_pre))
            o_t = T.nnet.softmax(T.dot(V, s_t))
            return [o_t[0], s_t]
        [o, s], updates = theano.scan(fn=forward_prop_step, sequences=x,
                                      outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
                                      non_sequences=[U, V, W], truncate_gradient=self.bptt_truncate,
                                      strict=True)
        # get the index of highest score
        prediction = T.argmax(o, axis=1)
        # cross-entropy
        o_error = T.nnet.categorical_crossentropy(o, y)

        # gradients
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)

        # assign functions
        self.forward_propagation = theano.function(inputs=[x], outputs=o)
        self.predict = theano.function(inputs=[x], outputs=prediction)
        self.ce_error = theano.function(inputs=[x, y], outputs=o_error)
        self.bptt = theano.function(inputs=[x, y], outputs=[dU, dV, dW])

        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function(inputs=[x, y, learning_rate], outputs=[], updates=
                                        [
                                            (self.U, self.U - learning_rate * dU),
                                            (self.V, self.V - learning_rate * dV),
                                            (self.W, self.W - learning_rate * dW)
                                        ])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(num_words)


def gradient_check_theano(model, x, y, h=0.001, error_thershold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt()
    # list of all parameters we want to check
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # get the actual parameter value from the model
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameters %s with size %d." % (pname, np.prod(parameter.shape))
        # iterate over each element of the parameter matrix,
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # save the original value so we can reset it later
            original_value = parameter[ix]

            # estimate the gradient using ((f(x + h) - f(x - h))/(2 * h))
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x], [y])

            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x], [y])

            estimated_gradient = (gradplus - gradminus) / (2 * h)

            parameter[ix] = original_value
            parameter_T.set_value(parameter)

            # the gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]

            # calculate the relative error: (|x-y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient)
                                                                               + np.abs(estimated_gradient))
            # if the error is too large fail the gradient check
            if relative_error > error_thershold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss %f" % gradminus
                print "Estimated gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameters %s passed." % pname





