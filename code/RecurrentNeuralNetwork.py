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
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
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
            # Note that we are indexing U by x_t . This is the same as multiplying U with a one-hot vector.
            # here x_t is a real number, i.e. 34
            s_t = T.tanh(U[:, x_t] + W.dot(s_t_pre))
            o_t = T.nnet.softmax(V.dot(s_t))
            # print s_t.shape, type(s_t)
            # print "o_t[0]: ", o_t[0], o_t.shape, type(o_t)
            # print s_t.eval()
            # print o_t.eval()
            return [o_t[0], s_t]               # why 0_t[0] ?
        [o, s], updates = theano.scan(fn=forward_prop_step, sequences=x,
                                      outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
                                      non_sequences=[U, V, W], truncate_gradient=self.bptt_truncate,
                                      strict=True)
        # get the index of highest score
        prediction = T.argmax(o, axis=1)
        # cross-entropy
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

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


def save_model_parameters_theano(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile


def load_model_parameters_theano(path, model):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    model.hidden_dim = U.shape[0]
    model.word_dim = U.shape[1]
    model.U.set_value(U)
    model.V.set_value(V)
    model.W.set_value(W)
    print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])


def train_with_sgd(model, X_train, Y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    """

    :param model: an RNN object
    :param X_train: a set of sentences and every sentence is tokenized, i.e. [[0, 179, 341, 416]]
    :param Y_train: same as X_train but shift by one position, i.e. [[179, 341, 416, 1]]
    :param learning_rate: float , learning_rate
    :param nepoch:iterate numbers
    :param evaluate_loss_after: every several epochs to evaluate loss
    :return:
    """
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # optionally evaluate the loss
        if epoch % evaluate_loss_after == 0:
            loss = model.calculate_loss(X_train, Y_train)
            losses.append((num_examples_seen, loss))

            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)

            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate = learning_rate * 0.5
                print "setting learning rate to %f " % learning_rate
            sys.stdout.flush()

            # saving model parameters
            save_model_parameters_theano("../data/rnn/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)

        for i in range(len(Y_train)):
            # for each training example.....
            # One SGD step
            model.sgd_step(X_train[i], Y_train[i], learning_rate)
            num_examples_seen += 1
        print "epoch %d done......." % epoch


def test_rnn():

    nltk.download("book")

    _VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
    _HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '100'))
    _LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
    _NEPOCH = int(os.environ.get('NEPOCH', '100'))
    _MODEL_FILE = os.environ.get('MODEL_FILE')

    vocabulary_size = _VOCABULARY_SIZE
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"

    # read data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file"
    with open('../data/reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # append SENTENCE_START, SENTENCE_END tokens
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences ." % (len(sentences))
    print ".......sample 4 sentences"
    print sentences[0]
    print sentences[1]
    print sentences[2]
    print sentences[3]

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    print ".......tokenized_sentences"
    print tokenized_sentences[0]
    print tokenized_sentences[1]
    print tokenized_sentences[2]
    print tokenized_sentences[3]

    # count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens. " % len(word_freq.items())
    # print word_freq
    # print word_freq.items()

    # get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    print "The top ten vocab: ", vocab[: 10]
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i)for i, w in enumerate(index_to_word)])

    print "Using vocabulary_size %d. " % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared '%d' times." % (vocab[-1][0], vocab[-1][1])

    # replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[: -1]] for sent in tokenized_sentences])
    Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    print "....sampling two train data:"
    print "X_train[0]: ", X_train[0]
    print "Y_train[0]: ", Y_train[0]

    np.random.seed(10)
    model = RNN(word_dim=vocabulary_size, hidden_dim=100)
    t1 = time.time()
    model.sgd_step(X_train[10], Y_train[10], _LEARNING_RATE)
    t2 = time.time()
    print "SGD Step time: %f milliseconds " % ((t2 - t1) * 1000.)

    if _MODEL_FILE !=None:
        load_model_parameters_theano(_MODEL_FILE, model)

    train_with_sgd(model, X_train[:100], Y_train[:100], nepoch=100, learning_rate=0.005, evaluate_loss_after=1)


if __name__ == '__main__':
    test_rnn()







