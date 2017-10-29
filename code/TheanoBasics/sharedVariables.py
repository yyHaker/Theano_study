# -*- coding: utf-8 -*-
from theano import shared
import theano.tensor as T
import theano
# define the accumulator function
state = shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates=[(state, state+inc)])
# use the accumulator
print state.get_value()
print accumulator(1)
print state.get_value()
print accumulator(300)
print state.get_value()

# reset the state
state.set_value(-1)
accumulator(3)
print state.get_value()

# define more than one function to use the same shared variable
decrementor = theano.function([inc], state, updates=[(state, state-inc)])
decrementor(2)
print state.get_value()

# use a shared variable but do not want to use its value
fn_of_state = state * 2 + inc
foo = T.scalar(dtype=state.dtype)
# use the 'givens' parameter of 'function' which replaces a particular node in a graph
#  for the purpose of one particular function.
skip_shared = theano.function([inc, foo], fn_of_state, givens=[(state, foo)])
print skip_shared(1, 3)  # use 3 for the state, not state.value
print state.get_value()
