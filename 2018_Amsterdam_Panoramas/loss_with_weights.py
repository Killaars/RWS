from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.client import device_lib
from tensorflow.core.protobuf import config_pb2

from collections import defaultdict

import numpy as np
from distutils.version import StrictVersion
import os,sys

from keras import backend as K

py_all = all
py_any = any
py_sum = sum
py_slice = slice

# INTERNAL UTILS

# This is the default internal TF session used by Keras.
# It can be set manually via `set_session(sess)`.
_SESSION = None

# This dictionary holds a mapping {graph: learning_phase}.
# A learning phase is a bool tensor used to run Keras models in
# either train mode (learning_phase == 1) or test mode (learning_phase == 0).
_GRAPH_LEARNING_PHASES = {}

# This dictionary holds a mapping {graph: UID_DICT}.
# each UID_DICT is a dictionary mapping name prefixes to a current index,
# used for generating graph-specific string UIDs
# for various names (e.g. layer names).
_GRAPH_UID_DICTS = {}

# This boolean flag can be set to True to leave variable initialization
# up to the user.
# Change its value via `manual_variable_initialization(value)`.
_MANUAL_VAR_INIT = False

# This list holds the available devices.
# It is populated when `_get_available_gpus()` is called for the first time.
# We assume our devices don't change during our lifetime.
_LOCAL_DEVICES = None



_EPSILON = 1e-7

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.
    # Returns
        A float.
    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```
    """
    return _EPSILON



def weighted_categorical_crossentropy(y_true,y_pred):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    other = 9000
    weights = [0.1,25,other,0.3,other]
    #weights = [1/128217,1/1220,1/40,1/1595,1/40] Gives too much class 1 and too much class 2, even though there is no class 2 in the testimage

    weights = K.variable(weights)
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

