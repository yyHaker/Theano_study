# -*- coding: utf-8 -*-
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

"""
1. Brief Example
伪随机数
"""
# put random variables in your graph
srng = RandomStreams(seed=234)  # 种子
rv_u = srng.uniform((2, 2))
rv_n = srng.normal((2, 2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)  # not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
# get random numbers
# different value call  f() twice
f_val0 = f()
print f_val0
f_val1 = f()
print f_val1
# same value call g() twice
g_val0 = g()
print g_val0
g_val1 = g()
print g_val1
# a random variable is drawn at most once during any single function execution
print nearly_zeros()

"""
2. Seeding Streams
上述2个随机变量，可以全局设置同一个种子，也可以分别设定
"""
# 分别设置，使用.rng.set_value()函数
rng_val = rv_u.rng.get_value(borrow=True)  # get the rng for rv_u
rng_val.seed(89234)                                        # seeds the generator
rv_u.rng.set_value(rng_val, borrow=True)  # Assign back seeded rng

# 全局设置，使用.seed()函数
# srng.seed(902340)
print "--------------------------------------------------------------"
"""
3. Sharing Streams Between Functions
"""
state_after_v0 = rv_u.rng.get_value().get_state()  # 保存调用前的state
print nearly_zeros()  # this affects rv_u's generator
v1 = f()  # 第一个调用，之后state会变化
print v1
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)  # 为其state还原
rv_u.rng.set_value(rng, borrow=True)
v2 = f()
print v2      # v2!=v1 输出更新后state对应的随机数
v3 = f()
print v3    # v3=v1  再次更新又还原成原来的state了
v4 = f()
print v4    # v4!=v1 state再次变化后的随机数

print "---------------------------------------------------------"

"""
4. Coping Random State Between Theano Graphs
i"""
import theano
import numpy
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
class Graph():
    def __init__(self, seed=123):
        self.rng = RandomStreams(seed)
        self.y = self.rng.uniform(size=(1,))

g1 = Graph(seed=123)
f1 = theano.function([], g1.y)

g2 = Graph(seed=987)
f2 = theano.function([], g2.y)
# by default the two functions are out of sync
print f1()
print f2()


def copy_random_state(g1, g2):
    if isinstance(g1.rng, MRG_RandomStreams):
        g2.rng.rstate = g1.rng.rstate
    for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
        su2[0].set_value(su1[0].get_value())

# we now copy the state of the theano random number generators
copy_random_state(g1, g2)
print f1()
print f2()