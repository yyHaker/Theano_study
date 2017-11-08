# -*- coding: utf-8 -*-
"""
test the dot between numpy and theano.tensor
"""

import theano
import theano.tensor as T
import numpy as np

A1 = [[0.1, 0.2, 0.3],
     [0.2, 0.3, 0.4],
     [0.4, 0.5, 0.6]]
B1 = [[0.1],
     [0.2],
     [0.3]]
C1 = [[0.1, 0.2, 0.3]]  # 1x3
# numpy.dot
print "-----------numpy.dot-----------"
A1 = np.asarray(A1)
B1 = np.asarray(B1)
print A1.dot(B1)     # 3x1
print np.dot(A1, B1)  # 3x1
print np.dot(C1, A1)  # 1x3
# theano.dot
print "------------theano.dot T.dot()-----------"
A = T.matrix('A')
B = T.matrix('B')
C = T.dot(A, B)
f = theano.function(inputs=[A, B], outputs=C)
D1 = f(A1, B1)   # 3x1
print D1
print type(D1)
print D1.shape

E1 = f(C1, A1)  # 1x3
print E1
print type(E1)
print E1.shape
# print dict(initial=np.zeros(10))
print "---------T.nnet.softmax()----------------"
D = T.nnet.softmax(C)
f_softmax = theano.function(inputs=[A, B], outputs=D)
F1 = f_softmax(A1, B1)
print F1
print type(F1)
print F1.shape  # 3x1
# 由于T.dot(C1, A1)是1x3, 所以softmax(T.dot(C1, A1))是1x3
F2 = f_softmax(C1, A1)
print F2
print type(F2)
print F2.shape  # 1x3

