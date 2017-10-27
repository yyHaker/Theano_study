# -*- coding: utf-8 -*-
"""
Implement Auto-encoder using Theano.

This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.

   They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = s(Wx+b), parameterized by {W,b}. The resulting latent representation y
 is then mapped back to a "reconstructed" vectorz = s(W'y + b').
 The weight matrix W' can optionally be constrained such that W' = W^T,
 in which case the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).


 For the denosing autoencoder, during training, first x is corrupted into
 tilde{x}, where tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 tilde{x}), y = s(W*tilde{x} + b) and z=s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - sum[ x_k * log z_k + (1-x_k) *log( 1-z_k)]
 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
"""

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy
from LogisticRegression import load_data
import os
import timeit

from PIL import Image
from utils import tile_raster_images


class AutoEncoder(object):
    """
     The Auto-encoder class.
    """
    def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500,
                 w=None, bhid=None, bvis=None):
        """
        Initialize the auto-encoder class by specifying the number of visible units(the dimension d of the input),
        the number of hidden units(the dimension of the latent or hidden space) and the corruption level.
        :param numpy_rng: numpy.random.RandomState, number random generator to generate weights
        :param theano_rng: theano.tensor.shared_randomstreams.RandomStreams, theano random generator
        :param input: theano.tensor.TensorType, a symbolic description of the input
        :param n_visible:int, number of visible units
        :param n_hidden: number of hidden units
        :param w: theano.tensor.TensorType
        :param bhid: theano.tensor.TensorType, a set of bias values for hidden units
        :param bvis: theano.tensor.TensorType, a set of bias values for visible layer
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create  a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # initialize w and b
        if not w:
            initial_w = numpy.asarray(numpy_rng.uniform(low=-4*numpy.sqrt(6./(n_hidden+n_visible)),
                                                        high=4*numpy.sqrt(6./(n_hidden+n_visible)),
                                                        size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            w = theano.shared(value=initial_w, name='w', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX), borrow=True)
        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), borrow=True)

        self.w = w
        self.bhid = bhid
        self.bvis = bvis
        self.w_vis = self.w.T

        self.theano_rng = theano_rng

        # if no input is given, generate a variable representing the input
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.w, self.bhid, self.bvis]

    def get_corrupted_input(self, input, corruption_level):
        """ This function keeps "1 - corruption_level" entries of the inputs the same and zero-out
        randomly selected subset of size "corruption_level".

        The binomial function will produce an array of 0s and 1s where 1 has a probability of "1 - corruption_level"
        and 0 with "corruption_level".
              - first argument: the shape of random numbers that it should produce
              - second argument: the number of trials
              - third argument: the probability of success of any trial
        :param input:
        :param corruption_level:
        :return:
        """
        return self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """
        Compute the values of the hidden layer
        :param input:
        :return:
        """
        return T.nnet.sigmoid(T.dot(input, self.w) + self.bhid)

    def get_reconstructed_input(self, hidden):
        """
        Compute the reconstructed input given the values of the hidden layer.
        :param hidden:
        :return:
        """
        return T.nnet.sigmoid(T.dot(hidden, self.w_vis) + self.bvis)

    def get_cost_updates(self, corruption_level, learning_rate):
        """
        Computes the cost and the updates for  one training step
        :param corruption_level:
        :param learning_rate:
        :return:
        """
        tilde_x = self.get_corrupted_input(self.x, corruption_level)

        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        # sum over the size of a datapoint; if we are using minibatches, L will be a vector, with one entry
        # per example in minibatch
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - 1), axis=1)
        cost = T.mean(L)

        # compute the gradients of the cost
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
        return cost, updates


def sgd_AutoEncoders(learning_rate=0.1, training_epochs=15, dataset='mnist.pkl.gz', batch_size=20,
                     output_folder='A_plots'):
    """
     Tested on MNIST.
    :param training_rate: float,
    :param training_epochs: int, number of epochs used for training
    :param dataset: string, path to the picked dataset
    :param batch_size:
    :param output_folder:
    :return:
    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index  to a minibatch
    x = T.matrix('x')   # the data is presented as rasterized images

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # ------------------------------------------------------------
    # Build the model no corruption
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    ae = AutoEncoder(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=28 * 28, n_hidden=500)

    cost, updates = ae.get_cost_updates(corruption_level=0, learning_rate=learning_rate)

    train_ae = theano.function(inputs=[index], outputs=cost, updates=updates,
                               givens={
                                   x: train_set_x[index * batch_size: (index + 1) * batch_size]
                               })

    start_time = timeit.default_timer()
    # Training
    for epoch in range(training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_ae(batch_index))
        print "Training epoch %d, cost " % epoch, numpy.mean(c, dtype='float64')
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    print "The no corruption code for file " + os.path.split(__file__)[1] + " ran for %.2fm" % training_time/60.

    image = Image.fromarray(tile_raster_images(X=ae.w.get_value(borrow=True).T, img_shape=(28*28),
                                               tile_shape=(10, 10), tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

    # --------------------------------------------------------------------------------
    # Building the model corruption 30%
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # denoising auto-encoders
    da = AutoEncoder(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=28 * 28, n_hidden=500)

    cost, updates = da.get_cost_updates(corruption_level=0.3, learning_rate=learning_rate)

    train_da = theano.function(inputs=[index], outputs=cost, updates=updates,
                               givens={
                                   x: train_set_x[index * batch_size: (index + 1) * batch_size]
                               })
    # Training
    start_time = timeit.default_timer()

    for epoch in range(training_epochs):
        c = []  # cost list
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
        print "Training epoch %d, cost " % epoch, numpy.mean(c, dtype='float64')

    end_time = timeit.default_timer()
    training_time = end_time - start_time

    print "The 30% corruption code for file " + os.path.split(__file__)[1] + "run for %.2fm" % training_time/60.

    image = Image.fromarray(tile_raster_images(X=da.w.get_value(borrow=True).T, img_shape=(28, 28),
                                               tile_shape=(10, 10), tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')


if __name__ == '__main__':
    sgd_AutoEncoders()
