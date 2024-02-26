import matplotlib.pyplot as plt

import numpy as np
import torch
import tensorflow as tf
import math  # for constant pi


def binary_classification():
    print("binary_classification")

    # Supervised Learning Problem
    #  This means we have an x and y and we are trying to learn a function that uses x to approximate y.

    # There are some problems with the idea of a binary classification. For example, if I predict that everything
    #  above a threshold (say 50%) is true and everything below that threshold is false, then I am neglecting the
    #  quality of the model. For example if it is 49%, it is absolutely false and 51% is absolutely true. A way to
    #  handle this problem is by using the metric called the ROC AUC (receiving operator characteristic, area under the
    #  curve) metric.


def integral_calculus_fn():
    binary_classification()
