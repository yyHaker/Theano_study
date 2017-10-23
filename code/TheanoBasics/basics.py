# -*- coding: utf-8 -*-
import numpy as np
import theano.tensor as T
from theano import function

"""
1. Adding two Scalars
"""
# create the function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)

# use the function
print f(2, 3)
print np.allclose(f(16.3, 12.1), 28.4)

# use eval()
print np.allclose(z.eval({x: 13, y: 12}), 25)

print "-------------------------------------"
"""
2. Adding two Matrices
   types can be found here: http://deeplearning.net/software/theano_versions/0.9.X/tutorial/adding.html
"""
x1 = T.dmatrix('x')
y1 = T.dmatrix('y')
z1 = x1 + y1
f1 = function([x1, y1], z1)
print f1([[1, 2], [3, 4]], [[10, 20], [30, 40]])
print f1(np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]]))

"""
3. Exercise
"""
a = T.vector()
out = a + a ** 10
f = function([a], out)
print f([0, 1, 2])


