# -*- coding: utf-8 -*-
import theano
import theano.tensor as T

"""
coping functions
"""
# define the accumulator
state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates=[(state, state+inc)])
# increment the state
accumulator(100)
print state.get_value()
# use 'copy()' to create a similar accumulator but with its own internal state using the
# 'swap' parameter, which is a dictionary of shared variables to exchange
new_state = theano.shared(0)
new_accumulator = accumulator.copy(swap={state: new_state})
new_accumulator(100)
print new_state.get_value()
# We now create a copy with updates removed using the delete_updates parameter,
# which is set to False by default:
null_accumulator = accumulator.copy(delete_updates=True)
null_accumulator(400)  # no longer updated
print state.get_value()

