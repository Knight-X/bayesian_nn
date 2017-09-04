from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import six

from utils import analytical_kl
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils
import tensorflow.contrib.distributions as distributions

__all__ = ['fully_connected']


def _build_variable_getter(rename=None):
    """Build a model variable getter that respects scope getter and renames."""
    # VariableScope will nest the getters
    def layer_variable_getter(getter, *args, **kwargs):
        kwargs['rename'] = rename
        return _model_variable_getter(getter, *args, **kwargs)
    return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
    """Adds variable (or all its parts) to all collections with that name."""
    collections = utils.get_variable_collections(
        collections_set, collections_name) or []
    variables_list = [variable]
    if isinstance(variable, tf_variables.PartitionedVariable):
        variables_list = [v for v in variable]
    for collection in collections:
        for var in variables_list:
            if var not in ops.get_collection(collection):
                ops.add_to_collection(collection, var)


@add_arg_scope
def fully_connected(inputs,
                    num_outputs,
                    posterior_sampler=None,
                    log_q=None,
                    log_p=None,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):
    """Adds a fully connected layer to a Bayesian neural net.

    Single layer of fully-connected units where the weights follow a
    unit gaussian prior, and

    Args:
        inputs: A tensor of at least rank 2 and static value for the last
            dimension; i.e. `[batch_size, depth]`,
            `[None, None, None, channels]`.
        num_outputs: Integer or long, the number of output units in the layer.
        posterior_sampler: Object to sample a weight matrix from the
            approximate posterior distribution by calling method
            `posterior_sampler.sample()`. Use default setting when object is
            None.
        log_q: Function that returns a tensor of batch log-probabilities
            of the approximate posterior given at the sample.
        log_p: Function that returns a tensor of batch log-probabilities
            of the prior given at the sample.
        activation_fn: Activation function. The default value is a ReLU
            function. Explicitly set it to None to skip it and maintain a
            linear activation.
        scope: Optional scope for variable_scope.

    Returns:
       The tensor variable representing the result of the series of operations.

    Raises:
        ValueError: If x has rank less than 2 or if its last dimension is not
                    set.
    """

    if not isinstance(num_outputs, six.integer_types):
        raise ValueError(
            'num_outputs should be int or long, got %s.' % (num_outputs,))

    with variable_scope.variable_scope(
            scope, 'fully_connected', [inputs],
            reuse=reuse, custom_getter=layer_variable_getter) as sc:

        inputs = ops.convert_to_tensor(inputs)

        # TODO: create layer class for bayesian neural net

        if posterior_sampler is None:
            num_inputs = tf.shape(inputs)[1]
            unit_normal = distributions.Normal(0., 1.)

            mean = tf.get_variable('mean', shape=[num_inputs, num_outputs],
                                   initializer=weights_initializer)
            rho = tf.get_variable('rho', shape=[num_inputs, num_outputs],
                                  initializer=weights_initializer)
            stddev = tf.nn.softplus(rho)
            sample = mean + stddev * \
                unit_normal.sample([num_inputs, num_outputs])
            kl = log_normal(mean, stddev, sample) - log_p(sample)
        else:
            sample = posterior_sampler.sample()
            kl = log_q(sample) - log_p(sample)

        biases = tf.get_variable('biases', shape=[num_outputs],
                                 initializer=biases_initializer)
        outputs = inputs * sample + biases

    # Add variables to collections.
    _add_variable_to_collections(layer.kernel, variables_collections,
                                 'weights')
    if layer.bias is not None:
        _add_variable_to_collections(layer.bias, variables_collections,
                                     'biases')

    # Apply normalizer function / layer.
    if normalizer_fn is not None:
        if not normalizer_params:
            normalizer_params = {}
        outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return utils.collect_named_outputs(
        outputs_collections, sc.original_name_scope, outputs)
