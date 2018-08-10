#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 2:09:22

@author: Wes Caldwell <caldwellwg@gmail.com>

Noise-aware layer adopted from Sukhbaatar et al.
"""


from keras import backend as K
from keras.constraints import Constraint
from keras.engine.topology import Layer
import numpy as np

def TraceNormRegularizer(X):
    """Evaluates the trace (nuclear) norm of a Keras tensor"""

    return 0.01 * np.linalg.norm(K.eval(X), ord='nuc')

class IsProbabilityMatrix(Constraint):
    """Constrains the weight matrix to be a left stochastic matrix."""

    def __call__(self, w):
        w *= K.cast(K.greater_equal(w, 0.), K.floatx()) # Ensure non-negativity
        return w / (K.epsilon() + K.sum(w, axis=0, keepdims=True)) # Ensure columns sum to 1

class NoiseAwareLayer(Layer):
    """Keras layer implementing label-flip noise-aware learning (Sukhbaatar et al. 2015)"""

    def __init__(self, **kwargs):
        super(NoiseAwareLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Q = self.add_weight(name='Q',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='identity',
                                 regularizer=TraceNormRegularizer,
                                 constraint=IsProbabilityMatrix())
        super(NoiseAwareLayer, self).build(input_shape)

    def call(self, x):
        return K.in_train_phase(K.dot(x, self.Q), x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])
