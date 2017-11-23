from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import six

import numpy as np
import numpy.random as npr
import tensorflow as tf

import bayesian_nn as bnn


def build_toy_data(num=100):
    """Builds a toy 1-D regression with an underlying quadratic function."""

    def f(x):
        """Ground truth function."""
        return -0.9 * x**2 + 1.9 * x + 1.

    xs = [[npr.rand()] for _ in range(num)]
    ys = [[f(x)] x in xs]

    return np.array(xs), np.array(ys)


def main(iteration=1000):

    xs, ys = build_toy_data()

    # symbolic variables
    sy_x = tf.placeholder([None, 1], dtype=tf.float32)
    sy_y = tf.placeholder([None, 1], dtype=tf.float32)

    fc_1 = bnn.layers.Dense('fc_1', 100, 100, FactorizedGaussian(100, 100), FactorizedGaussian(100, 100, is_prior=True))
    fc_2 = bnn.layers.Dense('fc_2', 100, 100, FactorizedGaussian(100, 100), FactorizedGaussian(100, 100, is_prior=True))

    # two layer bayesian neural net
    h, kl_1 = fc_1(sy_x)
    p, kl_2 = fc_2(tf.nn.relu(h))

    elbo = -tf.reduce_sum((p - sy_y)**2 - kl_1 - kl_2)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(-elbo)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
    	
    	sess.run(init_op)

    	for i in range(iteration):
    		sess.run(train_op)

    		if i % 10:
    			stat = sess.run(elbo)
    			print ('iteration [%d/%d] loss %.4f' % (i, iteration, stat))


if __name__ == '__main__':
    main()
