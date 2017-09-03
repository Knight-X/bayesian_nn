# bayesian-nn
bayesian-nn is a lightweight *Bayesian neural network* library built on top of tensorflow to ease network training via *stochastic variational inference* (SVI). The library is intended to resemble [slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) and help avoid massive boilerplate code. The end goal is to facilitate speedy development of Bayesian neural net models in the case where multiple stacked layers are required.

**Note: This project is still under active development!**

## Installation
```bash
pip install bayesian-nn
```

## Usage
```python
import bayesian-nn as bnn
```

## How are Bayesian neural nets trained with SVI?
![](assets/bbb_demo.gif)

Bayesian neural networks are just like ordinary neural networks except that weights are given explicit prior distributions, and the inferred posterior distributions given the data are used to make predictions on new data points. In addition to help avoid overfitting, Bayesian neural nets also give predictive uncertainty.

When making predictions, the model takes in account all weight configurations, which could be approximated through Monte Carlo.

The posterior distributions of the weights could be approximated through variational inference, where the evidence/variational lower bound (or negative variational free energy) is optimized so that the KL-divergence between the approximate and true posterior is minimized.

## Layers
bayesian-nn primarily provides the user with the flexibility of stacking neural net layers where weights follow approximate posterior distributions.

Pre-implemented layers include:

Layer | bayesian-nn
------- | --------
FullyConnected | [bnn.fully_connected]()
Conv2d | [bnn.conv2d]()
Conv2dTranspose (Deconv) | [bnn.conv2d_transpose]()
RNN | [bnn.rnn]()

## Features
The user can also further simplify boilerplate code through the following side features:

* [arg_scope](): allow users to define default arguments for specific operations within that scope.
* [repeat](): allow users to repeatedly perform the same operation with the same parameters.
* [stack](): allow users to perform the same operation with different parameters.

## References
