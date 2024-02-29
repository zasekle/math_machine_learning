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


def the_confusion_matrix():
    print("the_confusion_matrix")

    # The confusion matrix is a way to check if the algorithm is "confused" or not.

    #                        Confusion Matrix
    #                             actual y
    #               |       1        |        0       |
    #             -------------------------------------
    #             1 | True positive  | False positive |
    # predicted y -------------------------------------
    #             0 | False negative | True negative  |
    #             -------------------------------------


def the_receiver_operating_characteristic_roc_curve():
    print("the_receiver_operating_characteristic_ROC_curve")

    # This curve has a few steps involved with it. However, basically it represents how successfully the known data was
    #  matched by the machine learning algorithm. It looks at different thresholds for data and finds the percentage
    #  of negative cases that are incorrectly identified as positive vs the proportion of actual positive cases that are
    #  correctly identified by the model. Then it graphs them against each other and the area under this curve is the
    #  marker for how "correct" the machine learning model is.

    # Fundamentally, this is about graphing number that are true positives vs number true negatives. In other words how
    #  often the algorithm is correct. When it is the highest possible (1 vs 1), this means that it is correct every
    #  time.


def integral_calculus_fn():
    binary_classification()
    the_confusion_matrix()
    the_receiver_operating_characteristic_roc_curve()
