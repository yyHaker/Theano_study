# -*- coding: utf-8 -*-
"""
This tutorial introduces restrcited boltzmann machines (RBM) using theano.

Bolzmann Machines (BMs) are a particular form of energy-based model which contain
hidden variables. Restricted Bolzmann Machines further restrict BMs to those without
visible-visible and hidden-hidden connections.

"""
import timeit
import PIL.Image as Image
import numpy
import theano
import theano.tensor as T
import os
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import tile_raster_images
from LogisticRegression import load_data


class RBM(object):
    """Restricted Bolzmann Machines. (RBM)"""
    def __init__(self, input=None, n_visible=28 * 28, n_hidden=500, w=None, hbias=None, vbias=None,
                 numpy_rng=None, theano_rng=None):
        """
        RBM constructor. Define the parameters of the model along with basic operations for inferring
        hidden from visible (and vice-versa), as well as for performing CD updates.
        :param input: None for standalone RBMs or symbolic variable if RBM is part of a larger graph.
        :param n_visible: number of visible units
        :param n_hidden: number of hidden units
        :param w: None for standalone RBMs or symbolic variable pointing to a shared weights matrix in case
          RBM is part of a DBN network; in a DBN , the weights are shared between RBMs and layers of a MLP
        :param hbias: None for standalone RBMs or symbolic variable pointing to  a shared hidden units bias
          vector in case RBM is part of a different network
        :param vbias: None for standalone RBMs or a symbolic variable pointing to a shared visible units bias
        :param numpy_rng:
        :param theano_rng:
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a number generator
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)
        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # initial w and hbias, vbias, create theano shared variables and bias
        if w is None:
            initial_w = numpy.asarray(numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                size=(n_visible, n_hidden)
            ), dtype=theano.config.floatX)
        w = theano.shared(value=initial_w, name='w', borrow=True)

        if hbias is None:
            hbias = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX),
                                  name='hbias', borrow=True)
        if vbias is None:
            vbias = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX),
                                  name='vbias', borrow=True)

        # initialize input layer for standalone RBMs or layer0 for DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.w = w
        self.vbias = vbias
        self.hbias = hbias
        self.theano_rng = theano_rng

        self.params = [self.w, self.hbias, self.vbias]

    def propup(self, vis):
        """
        This function propagates the visible units activation upwards to the hidden units.

        :param vis:
        :return:
        """
        pre_sigmoid_activation = T.dot(vis, self.w) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_example):
        """
        This function infers state of hidden units given visible units.

        :param v0_example:
        :return:
        """
        # p(h|v), compute the activation of the hidden units given a sample of the visibles.
        pre_sigmoid_h1, h1_mean = self.propup(v0_example)

        # get a sample of the hiddens given their activation
        # 利用p(h|v)采样h
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        """
        This function propagates the hidden units activation down to the visible  units.
        :param hid:
        :return:
        """
        pre_sigmoid_activation = T.dot(hid, self.w.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        """
        This function infers state of visible units given hidden units.
        :param h0_sample:
        :return:
        """
        # p(v|h), compute the activation of the visible units given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)

        # get a sample of the visible given their activation
        v1_example = self.theano_rng.binomial(size=v1_mean.shape, n=1, p=v1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_example]

    def gibbs_hvh(self, h0_sample):
        """
        This function implements one step of Gibbs sampling, starting from the hidden state.
        :param h0_sample:
        :return:
        """
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """
        This function implements one step of Gibbs sampling, starting from the visible state.
        :param v0_sample:
        :return:
        """
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def free_energy(self, v_sample):
        """
        Function to compute the free energy.
        :param v_sample:
        :return:
        """
        wx_b = T.dot(v_sample, self.w) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def get_pseudo_likelihood_cost(self, updates):
        """
        Stochastic approximation to the pseudo-likelihood.
        :param updates:
        :return:
        """
        # index of bit i in expression p(xi|x-i)
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x-i
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy for given bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """
        Approximation to the reconstruction error.
        :param updates:
        :param pre_sigmoid_nv:
        :return:
        """
        cross_entropy = T.mean(T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                                     (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)), axis=1))
        return cross_entropy

    def get_cost_updates(self, learning_rate=0.1, persistent=None, k=1):
        """
        This function implements one step of CD-k or PCD-k.
        :param learning_rate: learning rate used to train the RBM.
        :param persistent: None for CD. For PCD, shared variables containing old state of Gibbs
        chain. This must be a shared variable of size (batch_size, number of hidden units).
        :param k: number of Gibbs steps to do in CD-k / PCD-k
        :return: a proxy for the cost and the updates dictionary.The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.
        """

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain: for CD, we use the newly generate hidden sample,
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # use "scan" to scan over the function that implements on gibbs step k times
        # the scan will return the entire Gibbs chain
        # the None are place holders, saying that chain_start is the initial state corresponding to the 6th
        ([pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates) = \
        theano.scan(self.gibbs_hvh, outputs_info=[None, None, None, None, None, chain_start], n_steps=k,
                    name="gibbs_hvh")

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        # we must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # construct the update dictionary
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(learning_rate, theano.config.floatX)
        if persistent:
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood cross-entropy is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates


def test_rbm(learning_rate=0.1, training_epochs=15, dataset='mnist.pkl.gz', batch_size=20, n_chains=20,
             n_samples=10, output_folder='rbm_plots', n_hidden=600):
    """
    Demonstrate how to train and afterwards sample from it using Theano.
    :param learning_rate:
    :param training_epochs:
    :param dataset:
    :param batch_size:
    :param n_chains: number of parallel Gibbs chains used to train the RBM
    :param n_samples: number of samples to plot for each chain
    :param output_folder:
    :param n_hidden:
    :return:
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar('index')  # index to a minibatch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden), dtype=theano.config.floatX),
                                     borrow=True)
    # construct the RBM class
    rbm = RBM(input=x, n_visible=28 * 28, n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(learning_rate=learning_rate, persistent=persistent_chain, k=15)

    """
    Train the RBM
    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    train_rbm = theano.function(inputs=[index], outputs=cost, updates=updates,
                                givens={
                                    x: train_set_x[index * batch_size: (index + 1) * batch_size]
                                }, name='train_rbm')

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in range(training_epochs):
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
        print "Training epoch %d, cost is " % epoch, numpy.mean(mean_cost)

        # plot filters after each training epoch
        plotting_start = timeit.default_timer()

        # construct image from the weight matrix
        image = Image.fromarray(tile_raster_images(X=rbm.w.get_value(borrow=True).T, img_shape=(28, 28),
                                                   tile_shape=(10, 10), tile_spacing=(1, 1)))
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print "Training took %f minutes " % (pretraining_time / 60.)

    """
    Sampling from the RBM
    """
    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(numpy.asarray(test_set_x.get_value(borrow=True)[test_idx: test_idx + n_chains],
                                                       dtype=theano.config.floatX))

    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean - field)
    # define a function that does 'plot_every' steps before returning the sample for plotting
    ([presig_hids, hid_mfs, hid_samples, presig_vis, vis_mfs, vis_samples], updates) = \
    theano.scan(rbm.gibbs_vhv, outputs_info=[None, None, None, None, None, persistent_vis_chain], n_steps=plot_every,
                name='gibbs_vhv')

    # add to updates the shared variable that takes care of our persistent chain
    updates.update({persistent_vis_chain: vis_samples[-1]})

    # construct the function that implements our persistent chain
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(inputs=[], outputs=[vis_mfs[-1], vis_samples[-1]], updates=updates, name='sample_fn')

    # create a space to store the image for plotting (we need to leave room for the tile_spacing as well)
    image_data = numpy.zeros((29 * n_samples + 1, 29 * n_chains -1), dtype='uint8')

    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print(' ... plotting sample %d' % idx)
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

        # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    # end-snippet-7
    os.chdir('../')


if __name__ == '__main__':
    test_rbm()