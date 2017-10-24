# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 400          # training sample size
feats = 784     # number of input variables

# generate a dataset: D=(input_variables, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# declare theano symbolic variables
x = T.dmatrix('x')
y = T.dvector('y')

# initialize the weight vector 'w' randomly
#
# this and the following bias variable b are shared so they keep their
# values between training iterations (updates)
w = theano.shared(rng.randn(feats), name="w")
# initialize the bias term
b = theano.shared(0., name="b")  # 注意是0.

print "initial model:"
print w.get_value()
print b.get_value()

# construct Theano expression graph
p_1 = 1/(1 + T.exp(-T.dot(x, w) - b))
prediction = p_1 > 0.5
xcent = -y * T.log(p_1) - (1-y) * T.log(1 - p_1)  # Cross-entropy loss function
cost = xcent.mean() + 0.01 * (w**2).sum()  # the cost to minimize
gw, gb = T.grad(cost, [w, b])  # compute gradient of the cost w.r.t weight vector w and bias term b

# compile
train = theano.function(inputs=[x, y], outputs=[prediction, cost],
                        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, cost = train(D[0], D[1])
    # every 1000 steps output some values
    if i % 100 == 0:
        print "iteration: steps %d,  cost = %g" % (i, cost)

print "Final model:"
print w.get_value()
print b.get_value()
print "target values for D: "
print D[1]
print "prediction on D: "
print predict(D[0])








