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
    def __init__(self, numpy_rng, theano_rng=None, n_ins=784, hidden_layers_sizes=[500, 500],
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

        :param train_set_x:theano.tensor.TensorType; shred var. that contains all datapoints used for training RBM
        :param batch_size: int; size of a minibatch
        :param k: number of Gibbs steps to do in CD-k / PCD-k
        :return:
        """
        # index to a minibatch
        index = T.lscalar('index')
        learining_rate = T.scalar('learning_rate')

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            # get cost and updates list using CD-k here for training each RBM
            cost, updates = rbm.get_cost_updates(learning_rate=learining_rate, persistent=None, k=k)

            # compile theno fumnction
            fn = theano.function(inputs=[index, theano.In(learining_rate, value=0.1)], outputs=cost, updates=updates,
                                 givens={
                                     self.x: train_set_x[batch_begin: batch_end]
                                 })
            pretrain_fns.append(fn)
        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """Generates a function 'train' that implements one step of finetuning, a function 'validate' that computes
        the error ona bacth form the validation set, and a function of 'test' that computes the error on a batch
        from the testing set.

        :param datasets: lists of pairs of theano.tensor.TensorType, It is a list that contain all datasets.
        :param batch_size: int, size of a minibatch
        :param learning_rate: float, learning rate used during finetune stage
        :return:
        """
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        # index toa minibatch
        index = T.lscalar('index')

        # compute the gradients with respect to the gradients
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(inputs=[index], outputs=self.finetune_cost, updates=updates,
                                   givens={
                                       self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                       self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
                                   })

        test_score_i = theano.function(inputs=[index], outputs=self.errors,
                                       givens={
                                           self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                           self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
                                       })

        valid_score_i = theano.function(inputs=[index], outputs=self.errors,
                                        givens={
                                            self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                            self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                                        })

        # create a function that scans the entire validation set
        def valid_socre():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_socre, test_score


def test_DBN(finetune_lr=0.1, pretraining_epoches=100, pretrain_lr=0.01, k=1, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=10):
    """
    Demonstrate how to train and test a Deep Belief Network.
    :param finetune_lr: float, learning rate used in the finetune stage
    :param pretraining_epoches: int , number of epoch to do pretraining
    :param pretrain_lr: float, learning rate to be used during pre-training
    :param k: int , number of Gibbs steps in CD/PCD
    :param training_epochs: int, maximal number of iterations to run the optimizer
    :param dataset: string, path the pickled dataset
    :param batch_size: int, the size od a minibatch
    :return:
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)

    print "......building the model"
    # construct the deep belief network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=28*28, hidden_layers_sizes=[1000, 1000, 1000], n_outs=10)

    print "....getting the pretraining functions"
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size, k=k)

    print "...pretraining the model"
    start_time = timeit.default_timer()
    for i in range(dbn.n_layers):
        # go through pretraining_epoches
        for epoch in range(pretraining_epoches):
            # go through the training set
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index, learning_rate=pretrain_lr))

            print "Pre-training layer %i, epoch %d, cost  " % (i, epoch), numpy.mean(c)
    end_time = timeit.default_timer()
    print "The pretraining code for file " + os.path.split(__file__)[1] + " ran for %.2fm " \
                                                                          % (end_time - start_time) / 60.

    # get the training, validation and testing function for the model
    print ".....getting the finetuning functions"
    train_fn, validate_model, test_model = dbn.build_finetune_functions(datasets=datasets, batch_size=batch_size,
                                                                        learning_rate=finetune_lr)

    print "...finetuning the model"

    # early stop parameters
    patience = 4 * n_train_batches
    patience_increase = 2.
    improvement_threshold = 0.995

    # 每迭代多少次validation一次
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_socre = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while epoch < training_epochs and not done_looping:
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                print "epoch %i, minibatch %i / %i, validation error %f %%" % (epoch, minibatch_index + 1,
                                                                               n_train_batches, this_validation_loss * 100.)

                # if we got the best validation score now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save the best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_socre = numpy.mean(test_losses, dtype='float64')
                    print "epoch %i, minibatch %i / %i, test error of best model %f %%" % (epoch, minibatch_index + 1,
                                                                                           n_train_batches, test_socre * 100.)
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print "Optimization complete with best validation score of %f %%, obtained at iteration %i, " \
          "with test performance %f %%" % (best_validation_loss * 100, best_iter + 1, test_socre * 100.)
    print "The fine tuning code for file " + os.path.split(__file__)[1] + " ran for %.2fm" %\
                                                                          (end_time - start_time) / 60.


if __name__ == "__main__":
    test_DBN()


