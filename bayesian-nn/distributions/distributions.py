from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import six

import tensorflow as tf


class AbstractDistribution:
    """Abstract base class for various distributions. Provides an interface."""

    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError

    def log_prob(self):
        """Called `log_prob` instead of `log_density` since we may also have
        discrete distributions."""
        raise NotImplementedError

    def __str__(self):
        return 'Abstract base class for all distributions'


class FactorizedGaussian(AbstractDistribution):
    """Factorized Gaussian distribution for a layer of weights."""

    def __init__(self, in_dims, ou_dims):
        pass


class MatrixVariateGaussian(AbstractDistribution):

    def __init__(self):
        raise NotImplementedError


class Gamma(AbstractDistribution):

    def __init__(self):
        raise NotImplementedError


class InverseGamma(AbstractDistribution):

    def __init__(self):
        raise NotImplementedError


class StudentT(AbstractDistribution):

    def __init__(self):
        raise NotImplementedError
