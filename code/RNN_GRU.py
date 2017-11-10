# -*- coding: utf-8 -*-
"""
implements the Recurrent Neural Network using GRU.

refering: http://www.wildml.com/2015/10/recurrent-
neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
"""
import numpy as np
import theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

import sys
import csv
import itertools
import nltk
from datetime import datetime

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"


class GRU(object):
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        """ Using the GRU to implement the Recurrent Neural Network
        :param word_dim: the size of our vocabulary
        :param hidden_dim: the size of hidden layer
        :param bptt_truncate:
        """
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # initialize the network parameters
        E = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((6, hidden_dim))
        c = np.zeros(word_dim)

        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))

        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c

        x = T.ivector('x')
        y = T.ivector('y')

        def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
            # word embedding layer
            x_e = E[:, x_t]

            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            # GRU Layer 2
            z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
            r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
            c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

            return [o_t, s_t1, s_t2]

        [o, s, s2], updates = theano.scan(fn=forward_prop_step, sequences=x, truncate_gradient=self.bptt_truncate,
                                          outputs_info=[None, dict(initial=T.zeros(self.hidden_dim)),
                                                        dict(initial=T.zeros(self.hidden_dim))])

        # get the index of the  biggest probability
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        # Total cost (could add regularization here)
        cost = o_error

        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

        # Assign functions
        self.predict = theano.function(inputs=[x], outputs=o)
        self.predict_class = theano.function(inputs=[x], outputs=prediction)
        self.ce_error = theano.function(inputs=[x, y], outputs=cost)
        self.bptt = theano.function(inputs=[x, y], outputs=[dE, dU, dW, db, dV, dc])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.sgd_step = theano.function(inputs=[x, y, learning_rate, theano.In(decay, value=0.9)],
                                        outputs=[], updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                                                             (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                                                             (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                                                             (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                                                             (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                                                             (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                                                             (self.mE, mE),
                                                             (self.mU, mU),
                                                             (self.mW, mW),
                                                             (self.mV, mV),
                                                             (self.mb, mb),
                                                             (self.mc, mc)])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for (x, y) in zip(X, Y)])

    def calculate_loss(self, X, Y):
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(num_words)


def train_with_sgd(model, X_train, Y_train, learning_rate=0.001, nepoch=1, decay=0.9,
                   callback_every=10000, callback=None):
    """
    train the RNN_GRU use SGD
    :param model: a GRU object
    :param X_train: a set of sentences and every sentence is tokenized, i.e. [[0, 179, 341, 416]]
    :param Y_train: same as X_train but shift by one position, i.e. [[179, 341, 416, 1]]
    :param learning_rate: float , learning rate
    :param nepoch: iteration numbers
    :param decay:
    :param callback_every:
    :param callback:
    :return:
    """
    num_examples_seen = 0
    for epoch in range(nepoch):
        # For each training example ....
        for i in np.random.permutation(len(Y_train)):
            # one SGD step
            model.sgd_step(X_train[i], Y_train[i], learning_rate, decay)
            num_examples_seen += 1
            #  Optionally do callback
            if callback and callback_every and num_examples_seen % callback_every == 0:
                callback(model, num_examples_seen)
    return model


def save_model_parameters_theano(model, outfile):
    """
    save the 'model' parameters to outfile.
    :param model: the model object
    :param outfile: the file path to be saved to
    :return:
    """
    np.savez(outfile, E=model.E.get_value(), U=model.U.get_value(), W=model.W.get_value(),
             V=model.V.get_value(), b=model.b.get_value(), c=model.c.get_value())
    print "Saved model parameters to %s. " % outfile


def load_model_parameters_theano(path, modelClass=GRU):
    """
    get the parameters from the "path" for the modelClass.
    :param path:
    :param modelClass:
    :return:
    """
    npzfile = np.load(path)
    E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d " % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)

    return model


def load_data(filename="../data/reddit-comments-2015-08.csv", vocabulary_szie=2000, min_sent_characters=0):

    word_to_index = []
    index_to_word = []

    # read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file...."
    with open(filename, 'rt') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode("utf-8").lower()) for x in reader])
        # filter sentences
        sentences = [s for s in sentences if len(s) > min_sent_characters]
        sentences = [s for s in sentences if "http" not in s]
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
    print "Parsed %d sentences ." % (len(sentences))

    print "sampled 5 sentences......."
    print sentences[0]
    print sentences[1]
    print sentences[2]
    print sentences[3]
    print sentences[4]

    # Tokenized the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # count the word frequence
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique word tokens. " % len(word_freq.items())

    # get the most common words and build index_to_word and word_to_index vectors
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_szie - 2]
    print "Using vocabulary_size %d. " % vocabulary_szie
    print "The least frequent word in our vocabulary is '%s' and appeared %d times. " % (vocab[-1][0], vocab[-1][1])

    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    print "samples 10 vacab......."
    print sorted_vocab[:10]
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with  unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[: -1]] for sent in tokenized_sentences])
    Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    print "sampled train data......"
    print X_train[10]
    print Y_train[10]

    return X_train, Y_train, word_to_index, index_to_word


def generate_sentence(model, index_to_word, word_to_index, min_llength=5):
    # we start the sentence with the start token
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
    # repeat until we get an end token
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(new_sentence)[-1]
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return None
    if len(new_sentence) < min_llength:
        return None
    return new_sentence


def generate_sentences(model, n, index_to_word, word_to_index):
    """
    generate n sentences and print
    :param model:
    :param n:
    :param index_to_word:
    :param word_to_index:
    :return:
    """
    for i in range(n):
        sent = None
        while not sent:
            sent = generate_sentence(model, index_to_word, word_to_index)
        # print sentences
        sentence_str = [index_to_word[x] for x in sent[1: -1]]
        print " ".join(sentence_str)
        sys.stdout.flush()


def train_GRU(vocabulary_size=2000, embedding_dim=48, hidden_dim=128,
              model_output_file=None, input_data_file='../data/reddit-comments-2015-08.csv',
              print_every=25000, learning_rate=0.001, nepoch=20, decay=0.9):
    """

    :param vocabulary_size:
    :param embedding_dim:
    :param hidden_dim:
    :param model_output_file:
    :param input_data_file:
    :param print_every:
    :param learning_rate:
    :param nepoch:
    :param decay:
    :return:
    """
    print "set output file......"
    if not model_output_file:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
        model_output_file = "../data/GRU/" + "GRU-%s-%s-%s-%s.dat" % (ts, vocabulary_size, embedding_dim, hidden_dim)

    # load data
    print "load data........"
    X_train, Y_train, word_to_index, index_to_word = load_data(input_data_file, vocabulary_size)

    # build model
    print "build model........."
    model = GRU(vocabulary_size, hidden_dim=hidden_dim, bptt_truncate=-1)

    # print SGD step time
    print "get the SGD setp time........"
    t1 = time.time()
    model.sgd_step(X_train[10], Y_train[10], learning_rate)
    t2 = time.time()
    print "SGD step time: %f seconds" % (t2 - t1)
    sys.stdout.flush()

    # we do this every few examples to understand what's going on
    def sgd_callback(model, num_examples_seen):
        dt = datetime.now().isoformat()
        loss = model.calculate_loss(X_train[:10000], Y_train[:10000])
        print "\n %s (%d)" % (dt, num_examples_seen)
        print "----------------------------------------------------"
        print "Loss: %f" % loss
        generate_sentences(model, 10, index_to_word, word_to_index)
        save_model_parameters_theano(model, model_output_file)
        print "\n"
        sys.stdout.flush()

    print "begin training......."
    for epoch in range(nepoch):
        train_with_sgd(model, X_train, Y_train, learning_rate=learning_rate, nepoch=1, decay=0.9,
                       callback_every=print_every, callback=sgd_callback)
        print "epoch %d done. " % epoch


if __name__ == "__main__":
    train_GRU()










