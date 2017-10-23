# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
"""
setting a default value for an argument
"""
x, y = T.dscalars('x', 'y')
z = x + y
f = theano.function([x, theano.In(y, value=1)], z)
print f(33)
print f(33, 2)

"""
- Inputs with default values must follow inputs without default values (like Python’s functions)
- default values can be set positionally or by name
"""
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = theano.function([x, theano.In(y, value=1), theano.In(w, value=2, name='w_by_name')], z)
print f(33)
print f(33, w_by_name=1, y=0)
