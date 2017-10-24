# -*- coding: utf-8 -*-
import theano
import numpy as np
import theano.tensor as T

"""
1. Computing tanh(x(t).dot(W) + b) elementwise
"""
# defining the tensor variables
x = T.matrix('x')
w = T.matrix('w')
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, w)) + b_sym, sequences=x)
compute_elementwise = theano.function(inputs=[x, w, b_sym], outputs=results)

# test values
x = np.eye(2, dtype=theano.config.floatX)
print x
w = np.ones((2, 2), dtype=theano.config.floatX)
print w
b = np.ones(2, dtype=theano.config.floatX)
print b
b[1] = 2
print compute_elementwise(x, w, b)

# comparison with numpy
print np.tanh(x.dot(w) + b)

